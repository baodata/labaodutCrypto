"""Read completed GP artifacts; export provenance/accounting and signal readout.

Does not refit or select models, alter the run, or evaluate a new portfolio.
Writes a new audit directory. Outcome access is offline reporting only.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.evaluation.decision_data import DecisionData
from src.evaluation.predictor_lock import load_predictor_lock
from src.evaluation.signal_allocation import load_allocation_config
from src.models.signal_dynamics import SignalDynamics
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes, new_run_directory


def audit(args):
    run = args.run_dir.resolve()
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Audit a completed run; unfinished cases do not form a full report.')
    checks = {name: file_hashes([Path(p) if Path(p).is_absolute() else ROOT/p for p in manifest[name]], ROOT)
              == manifest[name] for name in ('source_hashes', 'config_hashes', 'data_hashes', 'fit_hashes')}
    if not all(checks.values()):
        raise ValueError(f'GP artifact/input/source provenance differs: {checks}')
    output = new_run_directory(run/'audits', 'readout')
    atomic_json(output/'audit_source.json', dict(source_hashes=file_hashes([Path(__file__)], ROOT),
        audited_run=str(run), run_protocol=manifest['decision_protocol_hash']))
    lock, bank = load_predictor_lock(run/'predictor_lock.json')
    data = DecisionData(bank, lock, load_allocation_config(args.allocation_config),
        args.raw_dir, args.assets_config, args.env_config, output)
    fits, rows, signal = {}, [], []
    for case in sorted((run/'cases').glob('*/*/*')):
        checkpoint = json.loads((case/'completed.json').read_text())
        if file_hashes([case/'metrics.json', case/'daily_returns.csv', case/'decisions.jsonl'], ROOT) != checkpoint['hashes']:
            raise ValueError(f'GP completed case was modified: {case}')
        financial = json.loads((case/'metrics.json').read_text())
        fold, phase, method = financial['fold_id'], financial['phase'], financial['method']
        with (case/'decisions.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (fold, row['fit_id'])
                if key not in fits:
                    fits[key] = SignalDynamics.load(run/'fits'/fold/row['fit_id']/'signal_dynamics.npz')
                model = fits[key]
                cutoff = pd.Timestamp(model.metadata['fit_cutoff'])
                decision = pd.Timestamp(row['decision_date'])
                label_end = pd.Timestamp(row['maximum_label_end_used'])
                target = np.asarray(row['target'])
                budget = row['budget']
                stock_mean = budget/len(data.core)
                lower = stock_mean*(1-manifest['config']['tilt_fraction'])
                upper = min(stock_mean*(1+manifest['config']['tilt_fraction']), data.cap)
                rows.append(dict(fold_id=fold, phase=phase, method=method,
                    decision_date=str(decision.date()), fit_id=row['fit_id'],
                    causal_cutoff=bool(label_end <= cutoff < decision),
                    target_violation=max(abs(target.sum()-1), abs(target[:-1].sum()-budget),
                        float(np.max(lower-target[:-1])), float(np.max(target[:-1]-upper)), 0.),
                    nav_reconciliation_error=abs(row['nav_reconciliation_error']),
                    cost_reconciliation_error=abs(row['transaction_cost']-row['commission']-row['slippage']-row['impact']),
                    qp_primal_residual=row['primal_residual'], convex_gap=row.get('convex_objective_gap', 0.),
                    numerical_repair=row.get('numerical_feasibility_repair_l1', 0.)))
                if method == 'gp_constrained':
                    p = data.calendar.get_loc(decision)
                    y = data.closing[p+1]/data.opening[p+1]-1
                    y -= y.mean()
                    f = np.asarray(row['factors'])
                    calibrated = model.loading @ f
                    raw = model.mean+model.scale*f[:-1]
                    def rank_ic(values):
                        ranks = pd.Series(values).rank().to_numpy()
                        outcomes = pd.Series(y).rank().to_numpy()
                        return float(np.corrcoef(ranks, outcomes)[0, 1]) if min(ranks.std(), outcomes.std()) > 1e-12 else 0.
                    signal.append(dict(fold_id=fold, phase=phase, decision_date=str(decision.date()),
                        fit_id=row['fit_id'], calibrated_rank_ic=rank_ic(calibrated), raw_h5_rank_ic=rank_ic(raw),
                        mapping_mse=float(np.mean((calibrated-y)**2)), zero_mapping_mse=float(np.mean(y*y))))
    frame, signals = pd.DataFrame(rows), pd.DataFrame(signal)
    report = dict(provenance=checks, decisions=len(frame), fits=len(fits),
        causal_cutoff_violations=int((~frame.causal_cutoff).sum()),
        max_target_violation=float(frame.target_violation.max()),
        max_nav_reconciliation_error=float(frame.nav_reconciliation_error.max()),
        max_cost_reconciliation_error=float(frame.cost_reconciliation_error.max()),
        max_qp_primal_residual=float(frame.qp_primal_residual.max()),
        max_convex_gap=float(frame.convex_gap.max()),
        max_numerical_repair=float(frame.numerical_repair.max()),
        test_evaluated=False, independent_alpha_confirmation=False)
    atomic_json(output/'audit_summary.json', report)
    atomic_csv(frame, output/'decision_audit.csv', index=False)
    atomic_csv(signals, output/'signal_validation_daily.csv', index=False)
    atomic_csv(signals.groupby(['fold_id', 'phase'])[
        ['calibrated_rank_ic', 'raw_h5_rank_ic', 'mapping_mse', 'zero_mapping_mse']].mean(),
        output/'signal_validation_summary.csv')
    print(json.dumps(report, indent=2), flush=True)
    print(f'GP audit artifacts: {output}', flush=True)
    if (report['causal_cutoff_violations'] or report['max_target_violation'] > 1e-7
            or report['max_qp_primal_residual'] > 1e-7
            or report['max_convex_gap'] > manifest['config']['solver']['qp_gap_tolerance']
            or max(report['max_nav_reconciliation_error'], report['max_cost_reconciliation_error']) > 1e-6):
        raise RuntimeError('GP audit thresholds not met; inspect the report.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--allocation-config', type=Path, default=ROOT/'configs/signal_allocation_neural.yaml')
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--assets-config', type=Path, default=ROOT/'configs/assets.yaml')
    parser.add_argument('--env-config', type=Path, default=ROOT/'configs/env.yaml')
    with threadpool_limits(limits=1):
        audit(parser.parse_args())


if __name__ == '__main__':
    main()
