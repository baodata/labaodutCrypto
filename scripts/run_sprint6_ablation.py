"""Run predefined Sprint 6 ablations sequentially, one change per run.

No test backtest. Each variant uses the same frozen fold/seed and records its
resolved quant YAML. Defaults select just the control. Full experiments are
explicitly selected, never started as a side effect of importing the script.
"""
from pathlib import Path
import argparse
import copy
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.utils.artifacts import atomic_json, new_run_directory


def variant_configs(base):
    baseline = copy.deepcopy(base)
    baseline['normalization']['enabled'] = False
    baseline['graph']['covariance_source'] = 'legacy'
    baseline['impact']['eta'] = 0.
    baseline['impact']['accounting'] = 'legacy'
    baseline['execution']['enabled'] = False
    baseline['reward']['mode'] = 'excess_return'
    variants = {'control':baseline,'gru_control':copy.deepcopy(baseline)}
    # All normalized variants share the original fit interval. Enforce a
    # rolling warm-up for every arm via matching train-start in the runner.
    for name in ('xs','rolling','hybrid'):
        config = copy.deepcopy(baseline)
        norm = config['normalization']
        norm['enabled'] = True
        norm['modes'] = {f:'static_train' for f in ('xs_rank_ret_20d','xs_rank_ret_60d','rsi_14')}
        if name in {'xs','hybrid'}:
            norm['modes'].update(ret_20d='xs_zscore',mom_12_1='xs_rank')
        if name in {'rolling','hybrid'}:
            norm['modes'].update(volatility_20d='rolling_zscore',vol_60d='rolling_zscore')
        variants[name] = config
    for method in ('sample','ewma','ledoit_wolf'):
        config = copy.deepcopy(baseline)
        config['risk']['method'] = method
        config['graph']['covariance_source'] = 'risk'
        variants['graph_'+method] = config
    config = copy.deepcopy(baseline)
    config['graph']['covariance_source'] = 'residual'
    variants['graph_residual'] = config
    config = copy.deepcopy(baseline)
    config['impact']['accounting'] = 'cash_funded'
    variants['cash_accounting_control'] = copy.deepcopy(config)
    config['impact']['eta'] = .1
    variants['impact'] = config
    config = copy.deepcopy(baseline)
    config['impact']['accounting'] = 'cash_funded'
    config['execution']['enabled'] = True
    config['execution']['speed_mode'] = 'cost_aware'
    variants['adaptive_execution'] = config
    for mode in ('fixed', 'efficiency_ratio', 'asymmetric'):
        config = copy.deepcopy(variants['adaptive_execution'])
        config['execution'].update(speed_mode=mode, cost_risk_brake=True)
        variants['execution_'+mode] = config
    config = copy.deepcopy(baseline)
    config['reward']['mode'] = 'differential_sharpe'
    variants['dsr'] = config
    variants['sector_marl'] = copy.deepcopy(baseline)
    config = copy.deepcopy(baseline)
    config['marl']['reward_mode'] = 'sector_active'
    variants['sector_active_reward'] = config
    return variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants',default='control')
    parser.add_argument('--seeds',default='7,17,27,37,47')
    parser.add_argument('--quant-config',type=Path,default=ROOT/'configs/quant_sprint6.yaml')
    parser.add_argument('--research-protocol',type=Path,default=ROOT/'configs/quant_research_protocol.yaml')
    parser.add_argument('--protocol-fold',default='development_2022')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--min-epochs',type=int,default=10)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--episodes-per-update',type=int,default=32)
    parser.add_argument('--ppo-iters',type=int,default=10)
    parser.add_argument('--hidden-dim',type=int,default=128)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args(argv)
    with args.quant_config.open(encoding='utf-8') as f:
        variants = variant_configs(yaml.safe_load(f))
    names = args.variants.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    if any(name not in variants for name in names) or len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError(f'Unique variants/seeds required. Variants: {list(variants)}')
    from src.training.quant_context import validate_quant_dependencies
    for name in names:
        validate_quant_dependencies(variants[name])
    root = args.run_dir or new_run_directory(ROOT/'models/runs','sprint6_ablation')
    if args.run_dir:
        root.mkdir(parents=True,exist_ok=False)
    summaries = []
    manifest = dict(status='running',variants=names,seeds=seeds,commands=[],evaluation_scope='validation')
    atomic_json(root/'ablation_manifest.json',manifest)
    try:
        for name in names:
            for seed in seeds:
                quant_path = root/f'{name}_seed{seed}.yaml'
                quant_path.write_text(yaml.safe_dump(variants[name],sort_keys=False),encoding='utf-8')
                module = 'src.training.train_marl' if name.startswith('sector_') else 'src.training.train_ppo'
                encoder = 'gru' if name.startswith('sector_') or name=='gru_control' else 'gnn'
                command = [sys.executable,'-m',module,'--encoder',encoder,'--seed',str(seed),
                    '--quant-config',str(quant_path),'--research-protocol',str(args.research_protocol),
                    '--protocol-fold',args.protocol_fold,'--evaluation-scope','validation',
                    '--run-dir',str(root/f'{name}_seed{seed}'),'--epochs',str(args.epochs),
                    '--min-epochs',str(args.min_epochs),'--patience',str(args.patience),
                    '--episodes-per-update',str(args.episodes_per_update),'--ppo-iters',str(args.ppo_iters),
                    '--hidden-dim',str(args.hidden_dim),'--device',args.device,
                    '--normalization-warmup',str(int(variants['rolling']['normalization']['window']))]
                manifest['commands'].append(command)
                atomic_json(root/'ablation_manifest.json',manifest)
                subprocess.run(command,cwd=ROOT,check=True)
                import pandas as pd
                table = pd.read_csv(root/f'{name}_seed{seed}'/'validation_metrics.csv',index_col=0)
                table.insert(0,'variant',name)
                table.insert(1,'seed',seed)
                summaries.append(table.reset_index())
                from src.utils.artifacts import atomic_csv
                combined = pd.concat(summaries,ignore_index=True)
                atomic_csv(combined,root/'validation_comparison.csv',index=False)
                numeric = [c for c in combined.select_dtypes('number') if c != 'seed']
                atomic_csv(combined.groupby(['variant','Strategy'])[numeric].agg(['mean','std']),
                    root/'validation_comparison_mean_std.csv')
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        raise
    finally:
        atomic_json(root/'ablation_manifest.json',manifest)


if __name__=='__main__':
    main()
