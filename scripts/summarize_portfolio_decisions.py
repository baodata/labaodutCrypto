"""Recompute a paired report from saved returns; never retrain or select policies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.evaluation.portfolio_decision import summarize_decisions
from src.utils.artifacts import atomic_json, atomic_csv, file_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New derived report directory; original run is preserved.")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Derived report directory already contains artifacts.")
    root = args.run_dir.resolve()
    lock = json.loads((root / "predictor_lock.json").read_text())
    bank_manifest = Path(lock["prediction_bank"]) / "prediction_bank_manifest.json"
    if hashlib.sha256(bank_manifest.read_bytes()).hexdigest() != lock["bank_manifest_sha256"]:
        raise ValueError("Bank protocol snapshot differs from the run lock.")
    metadata = json.loads(bank_manifest.read_text())
    metrics, daily = pd.read_csv(root / "metrics_by_fold.csv"), pd.read_csv(root / "daily_returns.csv")
    report = summarize_decisions(metrics, daily, metadata["protocol"], lock["horizon_sessions"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(report, args.output_dir / "paired_comparison.csv", index=False)
    atomic_json(args.output_dir / "manifest.json", dict(status="completed", source_run=str(root),
        predictor_lock_digest=lock["lock_digest"], policy_or_checkpoint_changes=False,
        transformation="paired_summary_only_float64_noise_tolerance_1e-12_decimal_returns",
        input_hashes=file_hashes([root / "metrics_by_fold.csv", root / "daily_returns.csv",
                                root / "predictor_lock.json", bank_manifest], ROOT),
        source_hashes=file_hashes([ROOT / "src/evaluation/portfolio_decision.py", Path(__file__)], ROOT)))
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
