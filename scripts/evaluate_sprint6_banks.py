"""Export Sprint 6 bank diagnostics, volatility forecasts and execution stress.

Uses development validation only, with predetermined stress grid. This is a
research report generator, not a model/hyperparameter selector. Fixed equal
weights are compared under the same execution controller and cost assumptions.
"""
from pathlib import Path
import copy
import sys
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.training.cli import build_parser
from src.training.config import resolve_training_config
from src.training.run_artifacts import configure_run_artifacts, finish_run_manifest
from src.training.data_preparation import prepare_data, close_price_matrix
from src.training.env_factory import create_env
from src.risk.covariance import build_covariance_bank, ensure_covariance_dependencies
from src.graph.dynamic_graph import build_dynamic_graph_sequence
from src.training.quant_context import record_graph_diagnostics
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.policy_evaluator import metric_summary
from src.utils.artifacts import atomic_csv, atomic_json


def run(args):
    resolve_training_config(args)
    if args.evaluation_scope != 'validation' or args.quant_config is None:
        raise ValueError('Supply --quant-config and --evaluation-scope validation.')
    # This report explicitly compares all three estimators, even when the YAML
    # selects sample or EWMA for the main bank.
    ensure_covariance_dependencies('ledoit_wolf')
    configure_run_artifacts(args,'sprint6_banks')
    with args.env_config.open(encoding='utf-8') as f:
        config = yaml.safe_load(f)['environment']
    beta = float(config['reward_beta'] if args.reward_beta is None else args.reward_beta)
    data = prepare_data(args,beta)
    full,processed,scaled,split,train,val,_,_,val_open,_,_,val_close,_,tickers = data
    dates = pd.DatetimeIndex(full.dates)
    prices = close_price_matrix(processed,tickers,full.dates)
    returns = pd.DataFrame(prices,index=dates,columns=tickers).pct_change(fill_method=None)
    risk_config = {k:v for k,v in args._quant_config.get('risk',{}).items() if k != 'enabled'}
    forecasts = []
    n = len(tickers)
    weights = np.full(n,1/n)
    val_start = dates.get_loc(pd.Timestamp(val.dates[0]))
    val_end = dates.get_loc(pd.Timestamp(val.dates[-1]))
    for method in ('sample','ewma','ledoit_wolf'):
        params = dict(risk_config,method=method)
        bank = build_covariance_bank(returns,**params,cache_dir=args._run_dir/'banks')
        atomic_csv(bank.diagnostics,args._run_dir/f'{method}_covariance_diagnostics.csv',index=False)
        for t in range(val_start,val_end):
            if not bank.ready[t]:
                continue
            realized = float(returns.iloc[t+1].to_numpy()@weights)
            variance = float(weights@bank.covariance[t]@weights)
            forecasts.append(dict(method=method,decision_date=str(dates[t]),outcome_date=str(dates[t+1]),
                predicted_daily_variance=variance,realized_squared_daily_return=realized**2,
                squared_variance_error=(variance-realized**2)**2))
    forecast_table = pd.DataFrame(forecasts)
    atomic_csv(forecast_table,args._run_dir/'validation_volatility_forecasts.csv',index=False)
    atomic_csv(forecast_table.groupby('method').squared_variance_error.agg(['mean','count']),
        args._run_dir/'validation_volatility_forecast_summary.csv')
    with args.model_config.open(encoding='utf-8') as f:
        gnn = yaml.safe_load(f).get('gnn',{})
    graph_bank = scaled.quant_context.get('graph_risk',scaled.quant_context.get('risk'))
    adjacency,mask = build_dynamic_graph_sequence(returns,full.dates,tickers,
        window_size=int(gnn.get('correlation_window',60)),graph_mode=str(gnn.get('graph_mode','knn')),
        top_k=int(gnn.get('top_k',5)),threshold=float(gnn.get('threshold_corr',.5)),
        sector_config_path=str(args.assets_config),covariance_bank=graph_bank)
    record_graph_diagnostics(args,adjacency,mask)
    lookback = int(config.get('lookback_window',20) if args.lookback is None else args.lookback)
    cash = float(1/(n+1) if args.matched_cash_ratio is None else args.matched_cash_ratio)
    target = np.append(np.full(n,(1-cash)/n),cash)
    rows = []
    # Fixed in source before observing results. Do not select the best stress.
    grid = dict(eta=[0.,.1,.3],nav=[100000.,10000000.],controller=[False,True])
    for eta in grid['eta']:
        for nav in grid['nav']:
            for controller in grid['controller']:
                context = dict(scaled.quant_context,config=copy.deepcopy(scaled.quant_context['config']))
                context['config']['impact']['eta'] = eta
                context['config']['impact']['accounting'] = 'cash_funded'
                context['config']['execution']['enabled'] = controller
                context['config']['reward']['mode'] = 'excess_return'
                env_config = dict(config,initial_balance=nav)
                from src.env.trading_env import TradingEnv
                env = TradingEnv(val,val_open,close_prices=val_close,environment_config=env_config,
                    history_tensor=scaled,history_start_index=val_start,lookback_window=lookback,
                    action_alpha=args.action_alpha,reward_beta=beta,quant_context=context)
                result = DeterministicBacktester(env).run_agent(lambda obs:target)
                row = dict(eta=eta,nav=nav,adaptive_execution=controller,accounting='cash_funded',
                    **metric_summary(result,float(config['risk_free_rate'])))
                row['cost_reconciliation_error'] = float(abs(result.daily_costs.sum()-
                    result.daily_commission.sum()-result.daily_slippage.sum()-result.daily_impact.sum()))
                rows.append(row)
    atomic_csv(pd.DataFrame(rows),args._run_dir/'validation_execution_stress.csv',index=False)
    atomic_json(args._run_dir/'stress_protocol.json',dict(grid=grid,cash_ratio=cash,action_alpha=args.action_alpha,
        accounting='cash_funded in every arm to isolate impact/controller effects',
        policy='fixed equal stock weights; no fitting or test evaluation',
        forecast='Close[t] covariance vs squared equal-weight simple return at Close[t+1]; no fees'))
    finish_run_manifest(args,'completed',evaluation_scope='validation',test_evaluated=False)
    print(f'Bank/stress artifacts: {args._run_dir}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except BaseException as exc:
        if hasattr(args,'_manifest'):
            finish_run_manifest(args,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
