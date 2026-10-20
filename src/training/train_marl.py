"""Sector independent PPO with one jointly executed portfolio (Sprint 6).

Fixed equal sector budgets; shared net reward is the default cooperative game.
No manager or learned global allocator is implied. Uses development validation
only. Each agent has its own policy, critic, GAE buffer and optimizer.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
import torch
import yaml
from src.training.cli import build_parser
from src.training.config import resolve_training_config
from src.training.run_artifacts import configure_run_artifacts, finish_run_manifest
from src.training.data_preparation import prepare_data
from src.training.env_factory import create_env
from src.training.ppo_core import PPOBuffer, PPOUpdater
from src.env.marl_env import MultiAgentTradingEnv
from src.env.orders import PortfolioOrder
from src.evaluation.backtester import DeterministicBacktester
from src.evaluation.policy_evaluator import metric_summary
from src.evaluation.metrics import block_bootstrap_excess_returns
from src.models.actor_critic import build_actor_critic
from src.models.policy_adapter import observation_vector, history_mask_from_obs
from src.utils.data_types import MarketDataTensor
from src.utils.config import AssetsConfig
from src.utils.artifacts import atomic_csv, atomic_json, rng_state, restore_rng_state
from src.evaluation.research_protocol import checkpoint_learning_gate


def local_action(model, obs, device, deterministic=False):
    with torch.no_grad():
        return model.act(torch.as_tensor(observation_vector(obs), device=device),
            deterministic=deterministic, history_mask=history_mask_from_obs(obs, device))


def sector_target(wrapper, cash):
    weights = np.zeros(len(wrapper.tickers)+1)
    for agent in wrapper.agents:
        indices = wrapper.agent_to_indices[agent]
        weights[indices] = (1-cash)/len(indices)/wrapper.num_agents
    weights[-1] = cash
    return weights


def evaluate_sector_policies(wrapper, models, device):
    for model in models.values():
        model.eval()
    def policy(obs):
        split = wrapper._split_observation(obs)
        weights = np.zeros(wrapper.global_env.N+1)
        for agent in wrapper.agents:
            action = local_action(models[agent], split[agent], device, True)[0].cpu().numpy()
            weights[wrapper.agent_to_indices[agent]] = action[:-1]/wrapper.num_agents
            weights[-1] += action[-1]/wrapper.num_agents
        return weights
    return DeterministicBacktester(wrapper.global_env).run_agent(policy)


def run(args):
    resolve_training_config(args)
    if args.evaluation_scope != 'validation':
        raise ValueError("Sprint 6 MARL requires --evaluation-scope validation (no historical test tuning).")
    if args.encoder not in {'gru','lstm','mlp'}:
        raise ValueError("Sector PPO supports gru/lstm/mlp; the shared multi-relation GNN remains the single-agent ablation.")
    if args.seeds or args.untrained_null_seeds:
        raise ValueError("Run sector seeds separately with --seed and a new --run-dir.")
    if min(args.epochs,args.episodes_per_update,args.ppo_iters,args.hidden_dim) <= 0 or not 0 < args.action_alpha <= 1:
        raise ValueError("Invalid MARL training parameters.")
    configure_run_artifacts(args, 'sector_marl')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with args.env_config.open(encoding='utf-8') as f:
        env_config = yaml.safe_load(f)['environment']
    with args.model_config.open(encoding='utf-8') as f:
        rl = yaml.safe_load(f)['rl_agent']
    beta = float(env_config['reward_beta'] if args.reward_beta is None else args.reward_beta)
    data = prepare_data(args, beta)
    full, _, scaled, split, train, val, _, train_open, val_open, _, train_close, val_close, _, tickers = data
    dates = pd.DatetimeIndex(full.dates)
    lookback = int(env_config.get('lookback_window',20) if args.lookback is None else args.lookback)
    groups = AssetsConfig(str(args.assets_config)).raw_config['agent_groups']
    reward_mode = args._quant_config.get('marl',{}).get('reward_mode','shared')
    scaling = args.reward_scaling
    def make_env(tensor, prices_open, prices_close):
        start = int(dates.get_loc(pd.Timestamp(tensor.dates[0])))
        env = create_env(tensor, prices_open, args.env_config, beta, reward_scaling=scaling,
            history_tensor=scaled, history_start_index=start, reward_baseline=args.reward_baseline,
            lookback_window=lookback, action_alpha=args.action_alpha, close_prices=prices_close)
        return MultiAgentTradingEnv(env, groups, tickers, str(args.assets_config), reward_mode)
    val_wrapper = make_env(val, val_open, val_close)
    from src.training.quant_context import observation_context_dim
    state_dim = 2 + observation_context_dim(scaled)
    models, optimizers, schedulers, configs = {}, {}, {}, {}
    lr = float(rl['learning_rate'] if args.learning_rate is None else args.learning_rate)
    kappa = rl.get('dirichlet_kappa') if args.dirichlet_kappa is None else args.dirichlet_kappa
    gamma = float(rl.get('gamma',.95) if args.gamma is None else args.gamma)
    for agent in val_wrapper.agents:
        count = len(val_wrapper.agent_to_indices[agent])
        configs[agent] = dict(encoder=args.encoder,n_assets=count,n_features=train.tensor.shape[2],
            lookback=lookback,action_dim=count+1,hidden_dim=args.hidden_dim,rnn_layers=args.rnn_layers,
            rnn_dropout=args.rnn_dropout,dirichlet_kappa=kappa,portfolio_context_dim=state_dim)
        models[agent] = build_actor_critic(**configs[agent]).to(device)
        optimizers[agent] = torch.optim.Adam(models[agent].parameters(), lr=lr)
        schedulers[agent] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[agent],args.epochs,eta_min=1e-6)
    updater = PPOUpdater(clip_ratio=float(rl.get('clip_ratio',.2)),
        target_kl=float(rl.get('target_kl',.03) if args.target_kl is None else args.target_kl),
        value_loss_coef=float(rl.get('value_loss_coef',.5)),
        entropy_coef=float(rl.get('entropy_coef',.01) if args.entropy_coef is None else args.entropy_coef),
        max_grad_norm=float(rl.get('max_grad_norm',.5)),
        value_clip_range=rl.get('value_clip_range') if args.value_clip_range is None else args.value_clip_range)
    risk_free = float(env_config['risk_free_rate'])
    # One cash sleeve per local action: the matched sector baseline uses the
    # same initial expected cash share averaged across sectors.
    default_cash = args._quant_config.get('marl',{}).get('sector_cash_ratio',1/(len(tickers)+1))
    cash = float(args.matched_cash_ratio if args.matched_cash_ratio is not None else default_cash)
    if not 0 < cash < 1:
        raise ValueError('Sector Dirichlet baseline requires interior cash allocation.')
    for agent, model in models.items():
        dimensions = configs[agent]['action_dim']
        concentration = configs[agent]['dirichlet_kappa']
        if concentration is None:
            raise ValueError('Sector matched initialization requires fixed Dirichlet kappa.')
        desired = np.append(np.full(dimensions-1,(1-cash)/(dimensions-1)),cash)
        probabilities = (concentration*desired-1)/(concentration-dimensions)
        if np.any(probabilities <= 0):
            raise ValueError('Increase kappa to represent requested sector cash prior.')
        with torch.no_grad():
            model.actor_head.weight.zero_()
            model.actor_head.bias.copy_(torch.as_tensor(np.log(probabilities),dtype=torch.float32,device=device))
    target = sector_target(val_wrapper, cash)
    baseline = DeterministicBacktester(val_wrapper.global_env).run_agent(lambda obs: target)
    baseline_metrics = metric_summary(baseline, risk_free)
    initial = evaluate_sector_policies(val_wrapper, models, device)
    initial_metrics = metric_summary(initial, risk_free)
    def score(metrics):
        if args.selection_metric in {'sharpe','excess_sharpe'}:
            return metrics['sharpe_ratio'] - (baseline_metrics['sharpe_ratio'] if args.selection_metric=='excess_sharpe' else 0)
        return metrics['cumulative_return'] - (baseline_metrics['cumulative_return'] if args.selection_metric=='excess_return' else 0)
    reference_score = best = score(initial_metrics)
    best_epoch = 0
    learning_gate = checkpoint_learning_gate(0,initial_metrics,initial_metrics,baseline_metrics,0.,getattr(args,'_research_protocol',None))
    recent = [best]*args.selection_window
    stale = 0
    logs = []
    checkpoints = args._run_dir/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    training_parameters = {k:getattr(args,k) for k in ('epochs','ppo_iters','episodes_per_update',
        'window_min_days','window_max_days','batch_size','microbatch_size','action_alpha',
        'selection_metric','selection_margin','selection_window','patience','min_epochs','seed',
        'normalization_warmup','reward_baseline','reward_beta','reward_scaling','target_kl','learning_rate','entropy_coef','gamma')}
    def payload(epoch):
        return dict(epoch=epoch,best_epoch=best_epoch,model_config=configs,
            model_state_dict={a:m.state_dict() for a,m in models.items()},
            optimizer_state_dict={a:o.state_dict() for a,o in optimizers.items()},
            scheduler_state_dict={a:s.state_dict() for a,s in schedulers.items()},
            resolved_split_dates=args._resolved_split_dates,accounting_precision="float64",
            split_dates=split.dates_info,training_parameters=training_parameters,
            feature_parameters=args._feature_parameters,scaler_params=args._scaler_params,feature_names=list(train.feature_names),tickers=tickers,
            sector_mapping={a:val_wrapper.agent_to_indices[a] for a in models},portfolio_context_fields=['reward_state_if_dsr','market_context_if_xs','observed_budget','target_budget'],quant_config=args._quant_config,
            rng_state=rng_state(),training_state=dict(best_score=best,recent=recent,stale=stale,logs=logs),
            learning_gate=learning_gate,run_manifest=args._manifest,reference_score=reference_score,
            reward_state_convention='reset_each_episode_explicit_observation',
            reference_validation_metrics=initial_metrics,
            checkpoint_status='untrained_fallback' if epoch==0 else 'trained')
    def save(path, epoch):
        tmp = path.with_suffix('.tmp')
        torch.save(payload(epoch),tmp)
        tmp.replace(path)
    start_epoch = 1
    if args.resume is not None:
        previous = args._resume_payload
        if previous['model_config'] != configs or previous['training_parameters'] != training_parameters:
            raise ValueError('Resume requires identical sector architecture/training parameters.')
        for agent in models:
            models[agent].load_state_dict(previous['model_state_dict'][agent])
            optimizers[agent].load_state_dict(previous['optimizer_state_dict'][agent])
            schedulers[agent].load_state_dict(previous['scheduler_state_dict'][agent])
        state = previous['training_state']
        best, best_epoch, recent, stale, logs = state['best_score'],previous['best_epoch'],state['recent'],state['stale'],state['logs']
        reference_score = previous['reference_score']
        initial_metrics = previous['reference_validation_metrics']
        learning_gate = previous['learning_gate']
        source_best = args.resume.parent/'best.pth'
        if source_best.resolve() != (checkpoints/'best.pth').resolve():
            shutil.copy2(source_best,checkpoints/'best.pth')
        start_epoch = int(previous['epoch'])+1
        restore_rng_state(previous['rng_state'])
    else:
        save(checkpoints/'best.pth',0)
        save(checkpoints/'last.pth',0)
    atomic_json(args._run_dir/'sector_contract.json',dict(tickers=tickers,groups=val_wrapper.agent_to_indices,
        fixed_sector_budget=1/val_wrapper.num_agents,reward_mode=reward_mode,matched_cash_ratio=cash,
        baseline='fixed equal sector budgets, equal weights within sector; same alpha/controller/costs',
        initial_validation=initial_metrics,matched_validation=baseline_metrics))
    length_max = min(args.window_max_days,len(train.dates))
    if not 3 <= args.window_min_days <= length_max:
        raise ValueError('Invalid sector rollout length.')
    for epoch in range(start_epoch,args.epochs+1):
        for model in models.values():
            model.train()
        specs = []
        for _ in range(args.episodes_per_update):
            length = int(np.random.randint(args.window_min_days,length_max+1))
            start = int(np.random.randint(len(train.dates)-length+1))
            specs.append((start,length))
        size = sum(length-1 for _,length in specs)
        buffers = {a:PPOBuffer(size,lookback*cfg['n_assets']*cfg['n_features']+cfg['n_assets']+1+state_dim,
            cfg['action_dim'],gamma=gamma,lam=float(rl.get('gae_lambda',.95)),history_lookback=lookback)
            for a,cfg in configs.items()}
        episode_means = []
        for episode,(start,length) in enumerate(specs):
            end = start+length
            tensor = MarketDataTensor(train.tensor[start:end].copy(),list(tickers),list(train.feature_names),list(train.dates[start:end]))
            wrapper = make_env(tensor,train_open[start:end],train_close[start:end])
            obs,_ = wrapper.reset(seed=args.seed+epoch*args.episodes_per_update+episode)
            done = False
            total = 0.
            while not done:
                sampled = {a:local_action(models[a],obs[a],device) for a in models}
                actions = {a:values[0].cpu().numpy().astype(np.float32) for a,values in sampled.items()}
                following,rewards,terminated,truncated,info = wrapper.step(actions)
                for a in models:
                    buffers[a].store(observation_vector(obs[a]),actions[a],rewards[a],
                        float(sampled[a][2].item()),float(sampled[a][1].item()),history_mask=obs[a]['history_mask'])
                total += float(np.mean(list(rewards.values())))
                obs = following
                done = terminated or truncated
            episode_means.append(total)
            for a in models:
                bootstrap = 0. if end >= len(train.dates) else float(local_action(models[a],obs[a],device,True)[2].item())
                buffers[a].finish_path(bootstrap)
        for agent in models:
            batch = {k:v.to(device) for k,v in buffers[agent].get().items()}
            stats = updater.update(models[agent],optimizers[agent],batch,train_iters=args.ppo_iters,
                batch_size=args.batch_size,microbatch_size=args.microbatch_size or args.batch_size)
            schedulers[agent].step()
            logs.append(dict(epoch=epoch,agent=agent,**stats,mean_joint_episode_reward=float(np.mean(episode_means))))
        result = evaluate_sector_policies(val_wrapper,models,device)
        metrics = metric_summary(result,risk_free)
        recent = (recent+[score(metrics)])[-args.selection_window:]
        smooth = float(np.mean(recent))
        learning_gate = checkpoint_learning_gate(epoch,metrics,initial_metrics,baseline_metrics,
            smooth-reference_score,getattr(args,'_research_protocol',None))
        if np.isfinite(smooth) and smooth > best+args.selection_margin and (not getattr(args,'_research_protocol',None) or learning_gate['passed']):
            best,best_epoch,stale = smooth,epoch,0
            save(checkpoints/'best.pth',epoch)
        elif epoch >= args.min_epochs:
            stale += 1
        for row in logs[-len(models):]:
            row.update(validation_sharpe=metrics['sharpe_ratio'],
                sharpe_delta_vs_matched_baseline=metrics['sharpe_ratio']-baseline_metrics['sharpe_ratio'],
                validation_score=score(metrics),validation_smoothed_score=smooth,
                smoothed_improvement_vs_initial=smooth-reference_score,learning_gate_passed=learning_gate["passed"])
        save(checkpoints/'last.pth',epoch)
        atomic_csv(pd.DataFrame(logs),args._run_dir/'training_log.csv',index=False)
        print(f"epoch {epoch:03d} | sector validation Sharpe={metrics['sharpe_ratio']:.4f} "
              f"| Δmatched={metrics['sharpe_ratio']-baseline_metrics['sharpe_ratio']:+.4f}",flush=True)
        if args.patience > 0 and epoch >= args.min_epochs and stale >= args.patience:
            break
    saved = torch.load(checkpoints/'best.pth',map_location=device,weights_only=False)
    for agent in models:
        models[agent].load_state_dict(saved['model_state_dict'][agent])
    if saved['epoch']==0:
        warnings.warn('Sector checkpoint is untrained; report it explicitly.',RuntimeWarning)
    outer = getattr(args,'_outer_validation_context',None)
    if outer:
        report_wrapper = make_env(outer['val_tensor'],outer['val_open'],outer['val_close'])
    else:
        report_wrapper = val_wrapper
    learned = evaluate_sector_policies(report_wrapper,models,device)
    benchmark = DeterministicBacktester(report_wrapper.global_env).run_agent(lambda obs: target)
    asset_cash = float(cash)
    asset_target = np.append(np.full(len(tickers),(1-asset_cash)/len(tickers)),asset_cash)
    asset_baseline = DeterministicBacktester(report_wrapper.global_env).run_agent(lambda obs: asset_target)
    results = {'Sector MARL':learned,'Matched sector baseline':benchmark,'Matched asset 1/N + cash':asset_baseline}
    metrics = DeterministicBacktester(report_wrapper.global_env).compare(results,risk_free)
    metrics.index.name='Strategy'
    metrics['checkpoint_epoch'] = np.nan
    metrics.loc['Sector MARL','checkpoint_epoch'] = saved['epoch']
    metrics['sharpe_delta_vs_matched_baseline'] = metrics['sharpe_ratio']-metrics.loc['Matched sector baseline','sharpe_ratio']
    metrics['sharpe_delta_vs_matched_asset_baseline'] = metrics['sharpe_ratio']-metrics.loc['Matched asset 1/N + cash','sharpe_ratio']
    metrics['matched_baseline'] = 'Matched sector baseline'
    metrics['checkpoint_status'] = 'none'
    metrics.loc['Sector MARL','checkpoint_status'] = saved['checkpoint_status']
    atomic_csv(metrics,args._run_dir/'validation_metrics.csv')
    report_dates = report_wrapper.global_env.market_tensor.dates[1:]
    daily = pd.DataFrame({name:r.daily_net_returns for name,r in results.items()},index=report_dates)
    daily.index.name='date'
    atomic_csv(daily,args._run_dir/'validation_daily_returns.csv')
    bootstrap = block_bootstrap_excess_returns(learned.daily_net_returns,benchmark.daily_net_returns,
        block_size=args.bootstrap_block_size,num_samples=args.bootstrap_samples,seed=args.seed)
    atomic_json(args._run_dir/'validation_bootstrap.json',bootstrap)
    atomic_json(args._run_dir/'validation_report.json',dict(evaluation_scope='validation',test_evaluated=False,
        learning_gate=saved['learning_gate'],decision='GO' if saved['learning_gate']['passed'] else 'NO-GO',
        evaluation_period='outer_development' if outer else 'selection_validation',
        bootstrap_vs_matched=bootstrap,matched_baseline='equal sector budgets plus same cash/alpha/controller/costs'))
    finish_run_manifest(args,'completed',best_epoch=int(saved['epoch']),evaluation_scope='validation',
        report_period='outer_development' if outer else 'checkpoint_selection_validation',
        split_dates=split.dates_info,sector_reward_mode=reward_mode,matched_cash_ratio=cash)
    print(metrics.to_string())
    print(f"Artifacts: {args._run_dir}")


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    args = parser.parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        if hasattr(args,'_manifest'):
            finish_run_manifest(args,'interrupted')
        raise
    except Exception as exc:
        if hasattr(args,'_manifest'):
            finish_run_manifest(args,'failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
