"""Causal GP research runner with frozen forecasts and a derived decision protocol."""
from pathlib import Path
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml

from src.env.orders import PortfolioOrder
from src.env.quadratic_cost import fit_quadratic_cost
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.decision_data import DecisionData, ForecastEpisode, ROOT
from src.evaluation.gp_allocator import reference_diagnostics
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.predictor_lock import load_predictor_lock, digest
from src.evaluation.signal_allocation import load_allocation_config, _holm_adjust
from src.models.gp_adapter import GPAdapter, GP_METHODS
from src.models.signal_dynamics import fit_signal_dynamics
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory
from src.utils.data_types import MarketDataTensor


def load_gp_config(path):
    from src.utils.config import resolve_config_path
    config = yaml.safe_load(resolve_config_path(path).read_text())
    expected = {'version', 'methods', 'risk_aversion', 'discount_factor', 'tilt_fraction',
                'mpc_horizon', 'calibration', 'cost', 'execution', 'solver'}
    sections = dict(calibration={'minimum_samples', 'ridge', 'max_persistence', 'covariance_method', 'jitter', 'oof_refit_sessions'},
        cost={'planning_nav', 'scenario_eta', 'multiplier', 'weight_grid', 'minimum_eigenvalue'},
        execution={'cash_ratio', 'ramp_alpha', 'participation_cap', 'no_trade_band', 'realized_impact_eta'},
        solver={'bellman_tolerance', 'bellman_max_iterations', 'qp_tolerance', 'qp_gap_tolerance', 'qp_max_iterations'})
    if (not isinstance(config, dict) or set(config) != expected or config['version'] != 'gp-research-v1'
            or config['methods'] != list(GP_METHODS)):
        raise ValueError('GP config must declare the full fixed comparison family.')
    for name, keys in sections.items():
        if not isinstance(config[name], dict) or set(config[name]) != keys:
            raise ValueError(f'Invalid GP section: {name}.')
    ints = [config['mpc_horizon'], config['calibration']['minimum_samples'], config['calibration']['oof_refit_sessions'],
            config['solver']['bellman_max_iterations'], config['solver']['qp_max_iterations']]
    if any(type(v) is not int or v < 1 for v in ints) or config['calibration']['minimum_samples'] < 10:
        raise ValueError('Invalid GP counts.')
    numeric = [config['risk_aversion'], config['discount_factor'], config['tilt_fraction']]
    numeric += [v for name in sections for k, v in config[name].items()
                if k not in {'covariance_method', 'weight_grid'}]
    numeric += config['cost']['weight_grid']
    if not np.isfinite(numeric).all() or min(numeric) < 0:
        raise ValueError('GP numeric settings must be finite and nonnegative.')
    c, e, s, cost = config['calibration'], config['execution'], config['solver'], config['cost']
    if (config['risk_aversion'] <= 0 or not 0 < config['discount_factor'] < 1
            or not 0 <= config['tilt_fraction'] <= 1 or not 0 < c['max_persistence'] < 1
            or not 0 <= e['cash_ratio'] < 1 or not 0 < e['ramp_alpha'] <= 1
            or not 0 < e['participation_cap'] <= 1 or c['jitter'] <= 0
            or min(s['bellman_tolerance'], s['qp_tolerance'], s['qp_gap_tolerance'], cost['planning_nav'],
                   cost['minimum_eigenvalue'], cost['multiplier']) <= 0
            or not cost['weight_grid'] or min(cost['weight_grid']) <= 0 or max(cost['weight_grid']) > 1
            or c['covariance_method'] not in {'sample', 'ewma', 'ledoit_wolf'}):
        raise ValueError('GP parameter out of bounds.')
    return config


class FoldCalibration:
    def __init__(self, data, bank, selection, config, output):
        self.data, self.bank, self.selection, self.config, self.output = data, bank, selection, config, output
        self.fold = next(f for f in data.protocol['folds'] if f['name'] == selection['fold_id'])
        self.panels = {}
        for phase in ('train_oof', 'inner'):
            panel = bank.panel(model_id=selection['model_id'], fold_id=selection['fold_id'],
                split=phase, horizon=selection['horizon_sessions'])
            positions = data.calendar.get_indexer(pd.to_datetime(panel.index))
            rows = bank.predictions
            rows = rows[(rows.fold_id == selection['fold_id']) & (rows.model_id == selection['model_id'])
                & (rows.split == phase) & (rows.horizon_sessions == selection['horizon_sessions'])]
            ids = rows.drop_duplicates('decision_date').set_index('decision_date').fit_id.reindex(panel.index)
            if ids.isna().any():
                raise ValueError('Missing forecast fit provenance.')
            self.panels[phase] = (positions, panel.to_numpy(), ids.astype(str).to_numpy())
        self.cache = {}
        existing = output/f"{selection['fold_id']}_refits.csv"
        self.fit_rows = pd.read_csv(existing).to_dict('records') if existing.exists() and existing.stat().st_size > 1 else []

    def get(self, phase, position):
        data, cfg = self.data, self.config
        if phase == 'train_oof':
            origin = int(self.panels['train_oof'][0][0])+cfg['calibration']['minimum_samples']+1
            bucket = (int(position)-origin)//cfg['calibration']['oof_refit_sessions']
            cutoff = origin+bucket*cfg['calibration']['oof_refit_sessions']-1
            phases = ('train_oof',)
        else:
            boundary = self.fold['train_end' if phase == 'inner' else 'inner_end']
            cutoff = int(data.calendar.searchsorted(pd.Timestamp(boundary), side='right'))-1
            phases = ('train_oof',) if phase == 'inner' else ('train_oof', 'inner')
        if cutoff >= position:
            raise ValueError('GP refit must precede current decision.')
        key = (phases, cutoff)
        if key in self.cache:
            return self.cache[key]
        fit_id = f"{'_'.join(phases)}_cutoff_{cutoff}"
        path = self.output/'fits'/self.selection['fold_id']/fit_id
        if (path/'adapter.json').exists():
            adapter = GPAdapter.load(path)
            if (adapter.config != cfg or adapter.dynamics.metadata['tickers'] != data.tensor.tickers
                    or pd.Timestamp(adapter.dynamics.metadata['fit_cutoff']) != data.calendar[cutoff]):
                raise ValueError('GP fit artifact configuration/cutoff mismatch.')
            self.cache[key] = adapter
            if fit_id not in {row['fit_id'] for row in self.fit_rows}:
                self.fit_rows.append(dict(fold_id=self.selection['fold_id'], fit_id=fit_id,
                    cutoff=str(data.calendar[cutoff]), maximum_label_end_used=adapter.dynamics.metadata['maximum_label_end_used'],
                    samples=adapter.dynamics.metadata['samples'], **adapter.reference.diagnostics))
            return adapter
        positions = np.concatenate([self.panels[p][0] for p in phases])
        forecasts = np.concatenate([self.panels[p][1] for p in phases])
        ids = np.concatenate([self.panels[p][2] for p in phases])
        keep = positions+1 <= cutoff
        positions, forecasts, ids = positions[keep], forecasts[keep], ids[keep]
        if not len(positions):
            raise ValueError('No matured GP calibration outcomes.')
        intraday = data.closing[positions+1]/data.opening[positions+1]-1
        total = data.closing[positions+1]/data.closing[positions]-1
        calibration = cfg['calibration']
        dynamics = fit_signal_dynamics(forecasts, intraday, total, positions, ids, data.calendar,
            cutoff=cutoff, tickers=data.tensor.tickers, min_samples=calibration['minimum_samples'],
            ridge=calibration['ridge'], max_persistence=calibration['max_persistence'],
            covariance_method=calibration['covariance_method'], jitter=calibration['jitter'])
        cost_positions = positions[data.liquidity.ready[positions].all(1)]
        if not len(cost_positions):
            raise ValueError('No ready past liquidity for GP cost fit.')
        c = cfg['cost']
        cost, cost_metadata = fit_quadratic_cost(data.liquidity.dollar_adv[cost_positions],
            data.liquidity.volatility[cost_positions], nav=c['planning_nav'], eta=c['scenario_eta'],
            weight_grid=c['weight_grid'], multiplier=c['multiplier'], minimum_eigenvalue=c['minimum_eigenvalue'])
        adapter = GPAdapter(dynamics, cost, cfg, data.core, fit_id)
        adapter.save(path)
        atomic_json(path/'quadratic_cost.json', dict(cost_metadata,
            latest_cost_observation=str(data.calendar[cost_positions[-1]])))
        atomic_json(path/'provenance.json', dict(selection=self.selection, fit_phases=phases,
            decision_cutoff=str(data.calendar[cutoff]), maximum_label_end_used=dynamics.metadata['maximum_label_end_used'],
            latest_factor_used=str(data.calendar[positions[-1]]),
            family_selection='offline_inner_choice_not_known_at_historical_OOF_dates'))
        self.fit_rows.append(dict(fold_id=self.selection['fold_id'], fit_id=fit_id,
            cutoff=str(data.calendar[cutoff]), maximum_label_end_used=dynamics.metadata['maximum_label_end_used'],
            samples=dynamics.metadata['samples'], **adapter.reference.diagnostics))
        self.cache[key] = adapter
        return adapter


def run_case(data, calibration, episode, phase, method, settings):
    start, end = episode.start, episode.end
    tensor = MarketDataTensor(data.tensor.tensor[start:end], data.tensor.tickers,
        data.tensor.feature_names, data.tensor.dates[start:end])
    context = dict(data.tensor.quant_context, config=dict(execution=dict(enabled=False),
        impact=dict(accounting='cash_funded', eta=settings['execution']['realized_impact_eta'])))
    env = TradingEnv(tensor, data.opening[start:end], close_prices=data.closing[start:end],
        environment_config=data.environment, history_tensor=data.tensor, history_start_index=start,
        action_alpha=data.alpha, quant_context=context)
    decisions, pending = [], {}
    timer = time.perf_counter()

    def act(obs, info):
        position = start+env.current_step
        if position not in episode.forecasts:
            pending.clear()
            return PortfolioOrder.hold()
        adapter = calibration.get(phase, position)
        order, diagnostic = adapter.propose(method, episode.forecasts[position][0], env.portfolio.weights.copy(),
            step=env.current_step, adv=data.liquidity.dollar_adv[position], nav=env.portfolio.portfolio_value,
            max_asset_weight=data.cap, horizon=episode.horizon)
        pending.update(diagnostic, decision_date=str(data.calendar[position]),
            maximum_label_end_used=adapter.dynamics.metadata['maximum_label_end_used'],
            previous_close_weights=env.portfolio.weights.tolist())
        return order

    def observe(info):
        if not pending:
            return
        row = dict(pending, execution_date=info['execution_date'],
            filled_weights=info['execution_weights'].tolist(), transaction_cost=info['transaction_cost'],
            commission=info['commission_cost'], slippage=info['slippage_cost'], impact=info['impact_cost'],
            turnover=info['turnover'], net_return=info['net_return'], cash_pnl=info['cash_pnl'],
            nav_reconciliation_error=float(info['asset_net_pnl'].sum()+info['cash_pnl']
                -(info['portfolio_value']-info['previous_value'])),
            shortfall_l1=float(np.abs(info['execution_weights']-np.asarray(pending['target'])).sum()))
        decisions.append(row)
        pending.clear()

    result = DeterministicBacktester(env).run_strategy(act, on_step=observe)
    financial = metric_summary(result, float(data.environment['risk_free_rate']))
    financial.update(mean_close_cash=float(result.weights_history[:-1, -1].mean()),
        max_nav_reconciliation_error=max(abs(r['nav_reconciliation_error']) for r in decisions),
        max_solver_primal_residual=max(r['primal_residual'] for r in decisions),
        max_convex_objective_gap=max(r.get('convex_objective_gap', 0.) for r in decisions),
        max_shortfall_l1=max(r['shortfall_l1'] for r in decisions), seconds=time.perf_counter()-timer)
    return financial, result, decisions


def gp_gate_report(metrics, daily, protocol):
    records = []
    outer = daily[daily.phase == 'outer']
    if outer.empty:
        return pd.DataFrame()
    # Four predeclared hypotheses. Duplicate aim_gp is a numerical ablation, not
    # a second independent opportunity to reject the same economic null.
    for candidate in ('gp_constrained', 'gp_mpc'):
        for reference in ('matched_1n', 'convex'):
            left = outer[outer.method == candidate].set_index(['fold_id', 'date']).sort_index()
            right = outer[outer.method == reference].set_index(['fold_id', 'date']).sort_index()
            if not left.index.equals(right.index):
                raise ValueError('GP paired returns need identical fold/session coverage.')
            stats = block_bootstrap_excess_returns(left.net_return.to_numpy(), right.net_return.to_numpy(),
                block_size=protocol['bootstrap']['block_size'], num_samples=protocol['bootstrap']['samples'],
                seed=protocol['bootstrap']['seed'], groups=left.index.get_level_values(0).to_numpy())
            a = metrics[(metrics.phase == 'outer') & (metrics.method == candidate)].set_index('fold_id')
            b = metrics[(metrics.phase == 'outer') & (metrics.method == reference)].set_index('fold_id')
            delta = a.sharpe_ratio-b.sharpe_ratio
            records.append(dict(candidate=candidate, reference=reference, mean_sharpe_delta=float(delta.mean()),
                positive_folds=int((delta > 1e-12).sum()),
                worst_drawdown_degradation=float((b.max_drawdown-a.max_drawdown).max()),
                annual_net_excess=stats['mean_excess_annualized'], ci_lower_95=stats['ci_lower_95'],
                ci_upper_95=stats['ci_upper_95'], p_value=stats['p_value_positive_excess']))
    table = pd.DataFrame(records)
    table['holm_p_value'] = _holm_adjust(table.p_value.to_numpy())
    gate = protocol['gate']
    table['passes_portfolio_gate'] = ((table.positive_folds >= gate['minimum_positive_folds'])
        & (table.annual_net_excess >= gate['minimum_annualized_net_excess'])
        & (table.ci_lower_95 > gate['minimum_ci_lower']) & (table.holm_p_value < .05)
        & (table.mean_sharpe_delta > gate['minimum_mean_sharpe_delta'])
        & (table.worst_drawdown_degradation <= gate['maximum_drawdown_degradation']))
    return table


def compare_gp_allocators(args):
    from src.utils.config import resolve_config_path
    args.gp_config = resolve_config_path(args.gp_config)
    args.allocation_config = resolve_config_path(args.allocation_config)
    args.env_config = resolve_config_path(args.env_config)
    args.assets_config = resolve_config_path(args.assets_config)
    config, allocation = load_gp_config(args.gp_config), load_allocation_config(args.allocation_config)
    if args.fold and args.evaluate_development_outer:
        raise ValueError('Outer gate requires the whole declared fold family, not a handpicked fold.')
    lock, bank = load_predictor_lock(args.predictor_lock)
    selections = [s for s in lock['selections'] if args.fold is None or s['fold_id'] == args.fold]
    if not selections:
        raise ValueError('Unknown GP fold.')
    if (config['execution']['cash_ratio'] != bank.metadata['protocol']['execution']['cash_ratio']
            or config['execution']['ramp_alpha'] != bank.metadata['protocol']['execution']['action_alpha']):
        raise ValueError('GP control ramp/core must match declared prediction protocol settings.')
    if any(s['family'] not in allocation['models'] or s['horizon_sessions'] not in allocation['horizons'] for s in selections):
        raise ValueError('Allocation config does not cover the locked predictor.')
    output = args.resume.resolve() if args.resume else (args.output_dir or new_run_directory(ROOT/'models/runs', 'gp_research'))
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('GP output exists; choose a new directory or --resume.')
    output.mkdir(parents=True, exist_ok=True)
    hashes = dict(source_hashes=file_hashes(list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/compare_gp_allocators.py'], ROOT),
        config_hashes=file_hashes([args.gp_config, args.allocation_config, args.env_config, args.assets_config], ROOT),
        data_hashes=bank.metadata['data_hashes'],
        runtime={name: version(name) for name in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl')})
    hashes['runtime']['python'] = platform.python_version()
    old = json.loads((output/'manifest.json').read_text()) if args.resume else None
    if old and (any(old[k] != v for k, v in hashes.items()) or old['config'] != config
                or old['folds'] != [s['fold_id'] for s in selections]
                or old['outer_evaluated'] != args.evaluate_development_outer
                or old['predictor_lock_digest'] != lock['lock_digest']):
        raise ValueError('Resume requires identical source/data/config/fold/scope/lock.')
    if old and old.get('fit_hashes') != file_hashes((output/'fits').rglob('*.*'), ROOT):
        raise ValueError('GP fit artifacts changed; resume refused.')
    manifest = dict(status='running', config=config, predictor_lock_digest=lock['lock_digest'],
        folds=[s['fold_id'] for s in selections], outer_evaluated=args.evaluate_development_outer,
        test_evaluated=False, period_status='reused_historical_development_not_independent_holdout',
        initialization='cash_100pct_common_deterministic_planned_equity_budget_ramp',
        overnight_alpha='zero_expected_cross_sectional_gap_payoff_gap_risk_and_PnL_retained',
        cost_scenario='OHLCV_proxy_not_fill_calibration', resume_policy='restart_unfinished_case_from_cash_no_partial_NAV_stitching',
        **hashes)
    atomic_json(output/'manifest.json', manifest)
    atomic_json(output/'predictor_lock.json', lock)
    decision_protocol = dict(version='gp-decision-v1', source_prediction_protocol=bank.metadata['protocol_hash'],
        config=config, hypotheses=[(a, b) for a in ('gp_constrained', 'gp_mpc') for b in ('matched_1n', 'convex')],
        differences_from_prediction_protocol='final_Close_proposal_common_budget_ramp_no_second_alpha_quadratic_planning_realized_impact_scenario')
    decision_protocol['hash'] = digest(decision_protocol)
    atomic_json(output/'decision_protocol.json', decision_protocol)
    print(f'GP artifacts: {output}', flush=True)
    metrics, daily = [], []
    config = dict(config)
    try:
        atomic_json(output/'reference_numerical_acceptance.json', reference_diagnostics())
        data = DecisionData(bank, lock, allocation, args.raw_dir, args.assets_config, args.env_config, output)
        config['_linear_cost_rate'] = data.cost_rate
        for selection in selections:
            calibration = FoldCalibration(data, bank, selection, config, output)
            phases = ('train_oof', 'inner', 'outer') if args.evaluate_development_outer else ('train_oof', 'inner')
            for phase in phases:
                episode = data.episode(selection, phase)
                if phase == 'train_oof':
                    origin = int(calibration.panels['train_oof'][0][0])+config['calibration']['minimum_samples']+1
                    episode = ForecastEpisode(max(episode.start, origin), episode.end, episode.horizon,
                        {p: v for p, v in episode.forecasts.items() if p >= origin})
                if not episode.forecasts:
                    raise ValueError('No GP dates left after shared calibration warm-up.')
                for method in config['methods']:
                    case = output/'cases'/selection['fold_id']/phase/method
                    case.mkdir(parents=True, exist_ok=True)
                    if args.resume and (case/'completed.json').exists():
                        checkpoint = json.loads((case/'completed.json').read_text())
                        if file_hashes([case/'daily_returns.csv', case/'decisions.jsonl', case/'metrics.json'], ROOT) != checkpoint['hashes']:
                            raise ValueError('Completed GP case artifacts changed.')
                        row = json.loads((case/'metrics.json').read_text())
                        day = pd.read_csv(case/'daily_returns.csv')
                    else:
                        print(f"GP {selection['fold_id']} {phase} {method}...", flush=True)
                        financial, result, decisions = run_case(data, calibration, episode, phase, method, config)
                        row = dict(fold_id=selection['fold_id'], phase=phase, method=method,
                            predictor=selection['family'], horizon_sessions=selection['horizon_sessions'], **financial)
                        day = pd.DataFrame(dict(fold_id=selection['fold_id'], phase=phase, method=method,
                            date=data.calendar[episode.start+1:episode.end].astype(str),
                            net_return=result.daily_net_returns, turnover=result.daily_turnover,
                            costs=result.daily_costs, cash=result.weights_history[1:, -1]))
                        atomic_json(case/'metrics.json', row)
                        atomic_csv(day, case/'daily_returns.csv', index=False)
                        temporary = case/'decisions.tmp'
                        with temporary.open('w') as stream:
                            for record in decisions:
                                stream.write(json.dumps(record, default=str)+'\n')
                        temporary.replace(case/'decisions.jsonl')
                        atomic_json(case/'completed.json', dict(hashes=file_hashes(
                            [case/'metrics.json', case/'daily_returns.csv', case/'decisions.jsonl'], ROOT)))
                    metrics.append(row); daily.append(day)
                    atomic_csv(pd.DataFrame(metrics), output/'metrics_by_fold.csv', index=False)
                    atomic_csv(pd.concat(daily, ignore_index=True), output/'daily_returns.csv', index=False)
            atomic_csv(pd.DataFrame(calibration.fit_rows), output/f"{selection['fold_id']}_refits.csv", index=False)
        table = pd.DataFrame(metrics)
        combined = pd.concat(daily, ignore_index=True)
        paired = gp_gate_report(table, combined, bank.metadata['protocol'])
        if not paired.empty:
            atomic_csv(paired, output/'paired_comparison.csv', index=False)
        atomic_json(output/'gate_report.json', dict(outer_evaluated=args.evaluate_development_outer,
            independent_alpha_confirmation=False, hypotheses=paired.to_dict('records')))
        manifest.update(status='completed', cases=len(table), decision_protocol_hash=decision_protocol['hash'])
        print(table[['fold_id', 'phase', 'method', 'sharpe_ratio', 'total_costs', 'mean_close_cash']].to_string(index=False), flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        manifest['fit_hashes'] = file_hashes((output/'fits').rglob('*.*'), ROOT)
        atomic_json(output/'manifest.json', manifest)
    return output
