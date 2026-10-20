"""Compare Ridge/momentum/reversal using matched 1/N, bounded tilts and costs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.prediction_research import run_prediction_research
from src.evaluation.signal_allocation import compare_signal_allocators
from src.utils.artifacts import new_run_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-bank", type=Path, default=None)
    parser.add_argument("--prediction-config", type=Path, default=ROOT / "configs/prediction_bank.yaml")
    parser.add_argument("--research-protocol", type=Path, default=ROOT / "configs/quant_research_protocol.yaml")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/signal_allocation.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output_dir or new_run_directory(ROOT / "models/runs", "signal_allocation")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory already contains artifacts; choose a new run.")
    bank_dir = args.prediction_bank
    if bank_dir is None:
        built_bank, bank_dir = run_prediction_research(args.research_protocol, args.prediction_config,
            args.raw_dir, args.assets_config, output / "prediction_bank")
        del built_bank  # avoid retaining a second full bank during the comparison
    compare_signal_allocators(bank_dir, args.raw_dir, args.assets_config, args.env_config,
                              args.allocation_config, output / "comparison")


if __name__ == "__main__":
    main()
