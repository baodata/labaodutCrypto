"""Deterministic OOF training windows, independent of rollout RNG and inner scores."""
import numpy as np

from src.evaluation.decision_data import ForecastEpisode
from src.evaluation.policy_evaluator import metric_summary


def fixed_train_windows(training, count, days):
    available = training.end-training.start-1
    steps = min(days, available)
    if min(count, steps) <= 0:
        raise ValueError("Training probes require positive windows and decision days.")
    # Evenly spaced starts depend only on the training calendar; deduplicate
    # when a short history cannot support the configured number of windows.
    offsets = np.unique(np.linspace(0, available-steps, count, dtype=int))
    return [ForecastEpisode(training.start+int(start), training.start+int(start)+steps+1,
                            training.horizon, training.forecasts) for start in offsets]


def evaluate_train_windows(data, windows, policy, method, baselines, simulate):
    rows, rewards, tilts, turnovers = [], [], [], []
    for index, (window, baseline) in enumerate(zip(windows, baselines)):
        result, actions = simulate(data, window, method, policy)
        metrics = metric_summary(result, float(data.environment["risk_free_rate"]))
        reference = metric_summary(baseline, float(data.environment["risk_free_rate"]))
        excess = result.daily_net_returns-baseline.daily_net_returns
        targets = np.array([[row[f"target_{ticker}"] for ticker in data.tensor.tickers] for row in actions])
        rows.append(dict(window_id=index, **metrics,
            sharpe_delta_vs_matched=metrics["sharpe_ratio"]-reference["sharpe_ratio"],
            cumulative_return_delta_vs_matched=metrics["cumulative_return"]-reference["cumulative_return"],
            mean_daily_net_active_return=float(excess.mean()),
            mean_abs_target_tilt=float(np.abs(targets-data.core).mean())))
        rewards.extend(excess.tolist())
        tilts.extend(np.abs(targets-data.core).mean(axis=1).tolist())
        turnovers.extend(result.daily_turnover.tolist())
    summary = dict(train_probe_mean_sharpe=float(np.mean([r["sharpe_ratio"] for r in rows])),
        train_probe_mean_sharpe_delta_vs_matched=float(np.mean([r["sharpe_delta_vs_matched"] for r in rows])),
        train_probe_mean_return=float(np.mean([r["cumulative_return"] for r in rows])),
        train_probe_mean_daily_net_active_return=float(np.mean(rewards)),
        train_probe_mean_abs_target_tilt=float(np.mean(tilts)),
        train_probe_mean_daily_turnover=float(np.mean(turnovers)))
    return summary, rows
