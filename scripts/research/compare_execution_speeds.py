#!/usr/bin/env python3
"""
==============================================================================
TÊN SCRIPT: scripts/compare_execution_speeds.py
MỤC ĐÍCH:
    Mô phỏng thực thi lệnh tái cân bằng danh mục (execution controller) dưới các
    kịch bản tốc độ khác nhau (Immediate khớp ngay vs TWAP rải lệnh vs Trend-aware).
    Đo lường tác động của trượt giá (market impact), chi phí giao dịch và drift tiền mặt
    đối với tỷ suất Sharpe trên chuỗi tỷ trọng danh mục cố định (frozen target tape).

HƯỚNG DẪN CHẠY (CLI USAGE):
    python scripts/compare_execution_speeds.py \\
        --predictor-lock models/runs/predictor_lock.json

ĐẦU VÀO CẦN THIẾT (PREREQUISITES):
    - models/runs/predictor_lock.json
    - configs/research/execution_speed_research.yaml
    - configs/core/env.yaml, data/raw/*.parquet

ĐẦU RA SINH RA (OUTPUT ARTIFACTS):
    - models/runs/execution_speed_<timestamp>/
        + execution_comparison.csv: Bảng so sánh Sharpe, chi phí giữa các cơ chế
        + daily_returns.csv: Lợi nhuận ròng hàng ngày sau trừ phí
        + *_plans.csv: Chi tiết kế hoạch giải ngân và khớp lệnh từng phiên
==============================================================================
"""

from pathlib import Path
import argparse
import copy
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.env.orders import PortfolioOrder
from src.env.rebalance import ExecutionController
from src.env.trading_env import TradingEnv
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.cost_aware_allocator import cost_aware_target
from src.evaluation.decision_data import DecisionData
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.predictor_lock import load_predictor_lock
from src.evaluation.signal_allocation import load_allocation_config
from src.features.trend_efficiency import build_trend_efficiency_bank
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory
from src.utils.data_types import MarketDataTensor


def make_tape(data, episode):
    shadow = data.env(episode)
    tape = {}

    def act(obs, info):
        position = episode.start+shadow.current_step
        if position not in episode.forecasts:
            return PortfolioOrder.hold()
        target, _ = cost_aware_target(episode.forecasts[position][0], data.risk.covariance[position],
            shadow.portfolio.weights.copy(), horizon=episode.horizon, action_alpha=data.alpha,
            cash_ratio=data.cash, tilt_fraction=data.allocation['tilt_fraction'],
            max_asset_weight=data.cap, risk_aversion=data.allocation['risk_aversion'],
            cost_rate=data.cost_rate, cost_multiplier=data.allocation['cost_multiplier'])
        tape[position] = target.copy()
        return target

    DeterministicBacktester(shadow).run_strategy(act)
    return tape


def replay(data, episode, tape, execution, trend, *, matched):
    start, end = episode.start, episode.end
    tensor = MarketDataTensor(data.tensor.tensor[start:end], data.tensor.tickers,
        data.tensor.feature_names, data.tensor.dates[start:end])
    context = dict(data.tensor.quant_context, trend=trend,
        config=dict(impact=dict(accounting='cash_funded', eta=0.), execution=execution))
    env = TradingEnv(tensor, data.opening[start:end], close_prices=data.closing[start:end],
        environment_config=data.environment, history_tensor=data.tensor,
        history_start_index=start, lookback_window=1, action_alpha=data.alpha,
        reward_beta=1., quant_context=context)
    diagnostic_rows = []

    def act(obs, info):
        position = start+env.current_step
        action = (data.core_target if matched else tape[position]) if position in tape else PortfolioOrder.hold()
        return action

    result = DeterministicBacktester(env).run_strategy(act, on_step=diagnostic_rows.append)
    return result, diagnostic_rows


def run(args):
    from src.utils.config import resolve_config_path
    args.execution_config = resolve_config_path(args.execution_config)
    args.allocation_config = resolve_config_path(args.allocation_config)
    args.assets_config = resolve_config_path(args.assets_config)
    args.env_config = resolve_config_path(args.env_config)
    settings = yaml.safe_load(args.execution_config.read_text())
    if (not isinstance(settings, dict) or set(settings) != {'version', 'fold', 'common', 'variants'}
            or settings['version'] != 'execution-speed-research-v1'
            or not isinstance(settings['common'], dict) or not isinstance(settings['variants'], dict)
            or not settings['variants']
            or any(not isinstance(v, dict) for v in settings['variants'].values())):
        raise ValueError('Invalid execution research configuration.')
    variants = {name: dict(settings['common'], **overrides) for name, overrides in settings['variants'].items()}
    for execution in variants.values():
        controller = ExecutionController(execution)
        if not controller.replaces_action_alpha:
            raise ValueError('Replay requires explicit enabled fixed/ER/asymmetric speeds.')
    if 'fixed' not in variants or variants['fixed']['speed_mode'] != 'fixed' or variants['fixed']['cost_risk_brake']:
        raise ValueError('Keep a fixed-alpha control without a cost/risk brake.')
    allocation = load_allocation_config(args.allocation_config)
    lock, bank = load_predictor_lock(args.predictor_lock)
    selection = next((s for s in lock['selections'] if s['fold_id'] == settings['fold']), None)
    if selection is None:
        raise ValueError('Fold not present in predictor lock.')
    if selection['family'] not in allocation['models'] or selection['horizon_sessions'] not in allocation['horizons']:
        raise ValueError('Allocation configuration does not cover the locked predictor.')
    output = args.output_dir or new_run_directory(ROOT/'models/runs', 'execution_speeds')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', settings=settings, predictor_lock_digest=lock['lock_digest'],
        selection=selection, outer_evaluated=False, test_evaluated=False,
        purpose='execution_only_exploratory_replay_no_parameter_selection_or_alpha_gate',
        target_source='causal_convex_shadow_with_fixed_protocol_alpha_same_tape_all_variants',
        cost_model='cash_funded_commission_and_slippage_impact_eta_zero',
        source_hashes=file_hashes(list((ROOT/'src').rglob('*.py'))+[Path(__file__)], ROOT),
        config_hashes=file_hashes([args.execution_config, args.allocation_config, args.env_config, args.assets_config], ROOT),
        data_hashes=bank.metadata['data_hashes'])
    atomic_json(output/'manifest.json', manifest)
    print(f'Execution research artifacts: {output}', flush=True)
    try:
        data = DecisionData(bank, lock, allocation, args.raw_dir, args.assets_config, args.env_config, output)
        rows, daily = [], []
        trend_banks = {}
        for phase in ('train_oof', 'inner'):
            episode = data.episode(selection, phase)
            print(f'{phase}: freezing causal convex target tape...', flush=True)
            tape = make_tape(data, episode)
            atomic_csv(pd.DataFrame(list(tape.values()), index=[data.calendar[p] for p in tape],
                columns=list(data.tensor.tickers)+['cash']), output/f'{phase}_target_tape.csv')
            for name, execution in variants.items():
                window = execution.get('er_window', 10)
                if window not in trend_banks:
                    trend_banks[window] = build_trend_efficiency_bank(data.closing, data.calendar,
                        data.tensor.tickers, window=window)
                    trend = trend_banks[window]
                    np.savez_compressed(output/f'trend_er{window}.npz',
                        dates=np.asarray(data.calendar.astype(str)), tickers=np.asarray(data.tensor.tickers),
                        efficiency_ratio=trend.efficiency_ratio, ready=trend.ready)
                    atomic_json(output/f'trend_er{window}_metadata.json', trend.metadata)
                for strategy in ('matched_1n', 'convex_tape'):
                    result, diagnostics = replay(data, episode, tape, copy.deepcopy(execution),
                        trend_banks[window], matched=strategy == 'matched_1n')
                    row = metric_summary(result, float(data.environment['risk_free_rate']))
                    row.update(phase=phase, variant=name, strategy=strategy,
                        mean_close_cash=float(result.weights_history[:-1, -1].mean()),
                        hold_fraction=float(result.daily_hold.mean()),
                        total_commission=float(result.daily_commission.sum()),
                        total_slippage=float(result.daily_slippage.sum()),
                        total_impact=float(result.daily_impact.sum()))
                    row['cost_reconciliation_error'] = abs(row['total_costs']-row['total_commission']-row['total_slippage']-row['total_impact'])
                    rows.append(row)
                    daily.append(pd.DataFrame(dict(phase=phase, variant=name, strategy=strategy,
                        date=data.calendar[episode.start+1:episode.end], net_return=result.daily_net_returns,
                        turnover=result.daily_turnover, cost=result.daily_costs, cash=result.weights_history[1:, -1])))
                    plan_rows = []
                    for info in diagnostics:
                        plan = info.get('execution_controller')
                        if plan is not None:
                            for i, ticker in enumerate(data.tensor.tickers):
                                plan_rows.append(dict(execution_date=info['execution_date'], ticker=ticker,
                                    requested_alpha=plan['requested_alpha'][i], scheduled_alpha=plan['scheduled_alpha'][i],
                                    close_funded_alpha=plan['effective_alpha'][i], hold=bool(plan['hold_mask'][i])))
                    atomic_csv(pd.DataFrame(plan_rows), output/f'{phase}_{name}_{strategy}_plans.csv', index=False)
        metrics = pd.DataFrame(rows)
        for phase in metrics.phase.unique():
            part = metrics[metrics.phase == phase]
            fixed = part[(part.variant == 'fixed') & (part.strategy == 'matched_1n')].iloc[0]
            for index, row in part.iterrows():
                matched = part[(part.variant == row.variant) & (part.strategy == 'matched_1n')].iloc[0]
                metrics.loc[index, 'sharpe_delta_vs_same_controller_matched'] = row.sharpe_ratio-matched.sharpe_ratio
                metrics.loc[index, 'sharpe_delta_vs_fixed_matched'] = row.sharpe_ratio-fixed.sharpe_ratio
        atomic_csv(metrics, output/'execution_comparison.csv', index=False)
        atomic_csv(pd.concat(daily, ignore_index=True), output/'daily_returns.csv', index=False)
        manifest.update(status='completed', cases=len(metrics),
            target_tape_hashes=file_hashes(output.glob('*_target_tape.csv'), ROOT))
        print(metrics[['phase', 'variant', 'strategy', 'sharpe_ratio', 'mean_close_cash',
            'total_costs', 'sharpe_delta_vs_same_controller_matched']].to_string(index=False), flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        atomic_json(output/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor-lock', required=True, type=Path)
    parser.add_argument('--execution-config', type=Path, default=ROOT/'configs/research/execution_speed_research.yaml')
    parser.add_argument('--allocation-config', type=Path, default=ROOT/'configs/test/signal_allocation_neural.yaml')
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--assets-config', type=Path, default=ROOT/'configs/core/assets.yaml')
    parser.add_argument('--env-config', type=Path, default=ROOT/'configs/core/env.yaml')
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
