"""Sprint 6 data/config handoff, shared by PPO, MARL and evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.risk.covariance import build_covariance_bank, causal_residual_returns
from src.features.liquidity import build_liquidity_bank


def validate_quant_dependencies(config):
    """Only selected mechanisms may require optional backends."""
    risk = config.get('risk', {})
    needs_risk = (risk.get('enabled', False)
                  or config.get('graph', {}).get('covariance_source') in {'risk', 'residual'}
                  or config.get('execution', {}).get('enabled', False))
    if needs_risk:
        from src.risk.covariance import ensure_covariance_dependencies
        ensure_covariance_dependencies(risk.get('method', 'ledoit_wolf'))


def load_quant_config(args):
    path = getattr(args, 'quant_config', None)
    if path is None:
        return {}
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("quant-config must contain a mapping.")
    allowed = {
        'normalization': {'enabled','modes','window','min_periods','eps','clip_range','default_mode','preserve_market_context'},
        'risk': {'enabled','window','min_periods','method','decay','jitter'},
        'liquidity': {'enabled','window','min_periods'},
        'graph': {'covariance_source','residual_window','missing_benchmark'},
        'impact': {'eta','participation_warning','accounting'},
        'execution': {'enabled','no_trade_band','participation_cap','cost_aversion','risk_aversion',
                      'speed_mode','alpha_min','alpha_max','alpha_buy','alpha_sell','er_window','cost_risk_brake'},
        'reward': {'mode','reference','eta','epsilon','warmup','clip','scaling'},
        'marl': {'reward_mode','sector_cash_ratio'},
    }
    if set(config)-set(allowed):
        raise ValueError(f'Unknown quant config sections: {set(config)-set(allowed)}')
    for section,values in config.items():
        if not isinstance(values,dict) or set(values)-allowed[section]:
            raise ValueError(f'Unknown/invalid quant keys in {section}.')
        if 'enabled' in values and type(values['enabled']) is not bool:
            raise ValueError(f'{section}.enabled must be boolean.')
    if config.get('impact',{}).get('accounting','legacy') not in {'legacy','cash_funded'}:
        raise ValueError('Unknown execution accounting mode.')
    if config.get('graph',{}).get('missing_benchmark','error') != 'error':
        raise ValueError('Only explicit error on missing residual benchmark is supported.')
    if config.get('graph', {}).get('covariance_source', 'legacy') not in {'legacy', 'risk', 'residual'}:
        raise ValueError("Unknown graph covariance source.")
    if config.get('reward', {}).get('mode', 'excess_return') not in {'excess_return', 'differential_sharpe'}:
        raise ValueError("Unknown reward mode.")
    validate_quant_dependencies(config)
    from src.env.rebalance import ExecutionController
    ExecutionController(config.get('execution', {}))
    return config


def prepare_quant_context(args, processed, full_tensor, close_prices):
    config = load_quant_config(args)
    if not config:
        return None
    dates = pd.DatetimeIndex(full_tensor.dates)
    tickers = list(full_tensor.tickers)
    returns = pd.DataFrame(close_prices, index=dates, columns=tickers).pct_change(fill_method=None)
    context = dict(config=config, dates=list(dates), tickers=tickers)
    execution = config.get('execution', {})
    if execution.get('enabled', False) and execution.get('speed_mode') == 'efficiency_ratio':
        from src.features.trend_efficiency import build_trend_efficiency_bank
        context['trend'] = build_trend_efficiency_bank(close_prices, dates, tickers,
            window=execution.get('er_window', 10))
    risk = config.get('risk', {})
    graph = config.get('graph', {})
    if risk.get('enabled', False) or graph.get('covariance_source') in {'risk','residual'} or config.get('execution', {}).get('enabled', False):
        params = {k: v for k,v in risk.items() if k != 'enabled'}
        context['risk'] = build_covariance_bank(returns, **params,
            cache_dir=args._run_dir / 'banks' if hasattr(args,'_run_dir') else None)
    if config.get('liquidity', {}).get('enabled', False) or config.get('impact', {}).get('eta', 0) or config.get('execution', {}).get('enabled', False) or config.get('impact', {}).get('accounting') == 'cash_funded':
        params = {k:v for k,v in config.get('liquidity', {}).items() if k != 'enabled'}
        context['liquidity'] = build_liquidity_bank(processed, dates, tickers, **params)
    if graph.get('covariance_source') == 'residual':
        from src.utils.config import AssetsConfig
        benchmark = AssetsConfig(str(args.assets_config)).raw_config['benchmark']['ticker']
        frame = processed.get(benchmark)
        if frame is None:
            raise ValueError(f'Residual graph requires aligned benchmark {benchmark}; no silent fallback.')
        lookup = {str(c).lower().replace(' ', '_'): c for c in frame}
        if 'adj_close' not in lookup:
            raise ValueError('Residual benchmark needs Adj Close on same return basis.')
        benchmark_prices = frame[lookup['adj_close']].reindex(dates)
        if not np.isfinite(benchmark_prices).all() or (benchmark_prices <= 0).any():
            raise ValueError('Residual benchmark is missing/invalid on the stock trading calendar.')
        benchmark_returns = benchmark_prices.pct_change(fill_method=None)
        residual = causal_residual_returns(returns, benchmark_returns, window=int(graph.get('residual_window',60)))
        context['graph_risk'] = build_covariance_bank(residual, **params_for_risk(risk),
            cache_dir=args._run_dir / 'banks' if hasattr(args,'_run_dir') else None)
    elif graph.get('covariance_source') == 'risk':
        context['graph_risk'] = context['risk']
    if hasattr(args, '_run_dir'):
        from src.utils.artifacts import atomic_json, atomic_csv
        metadata = {'config': config, 'tickers': tickers, 'dates': [str(d) for d in dates]}
        for name in ('risk','graph_risk','liquidity','trend'):
            bank = context.get(name)
            if bank is not None:
                metadata[name] = bank.metadata
                payload = dict(dates=np.asarray([str(d) for d in dates]),tickers=np.asarray(tickers),ready=bank.ready)
                if hasattr(bank,'covariance'):
                    payload['covariance'] = bank.covariance
                elif hasattr(bank, 'efficiency_ratio'):
                    payload['efficiency_ratio'] = bank.efficiency_ratio
                else:
                    payload.update(dollar_adv=bank.dollar_adv,volatility=bank.volatility)
                bank_path = args._run_dir / f'{name}_bank.npz'
                temporary = bank_path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,**payload)
                temporary.replace(bank_path)
                if hasattr(bank,'diagnostics'):
                    atomic_csv(bank.diagnostics, args._run_dir / f'{name}_diagnostics.csv', index=False)
        atomic_json(args._run_dir / 'quant_context.json', metadata)
    return context


def params_for_risk(config):
    return {k:v for k,v in config.items() if k != 'enabled'}


def record_graph_diagnostics(args, adjacency, edge_mask):
    if not getattr(args, 'quant_config', None) or not hasattr(args, '_run_dir'):
        return
    from scipy.sparse.csgraph import connected_components
    from src.graph.multi_relation_graph import GRAPH_RELATIONS
    from src.utils.artifacts import atomic_csv
    rows = []
    previous = None
    for t, mask in enumerate(edge_mask.astype(bool)):
        union = np.any(mask,axis=0)
        changed = 0. if previous is None else float(np.count_nonzero(mask != previous)/max(np.count_nonzero(mask | previous),1))
        rows.append(dict(day_index=t,edge_churn=changed,
            connected_components=int(connected_components(union,directed=False,return_labels=False)),
            **{f'edges_{name}':int(mask[r].sum()) for r,name in enumerate(GRAPH_RELATIONS)}))
        previous = mask
    atomic_csv(pd.DataFrame(rows),args._run_dir/'graph_diagnostics.csv',index=False)


def observation_context_dim(tensor):
    context = tensor.quant_context or {}
    reward = context.get('config', {}).get('reward', {})
    return (3 if reward.get('mode') == 'differential_sharpe' else 0) + (context['market_context'].shape[-1] if 'market_context' in context else 0)
