"""Compare convex → PPO → sector MARL → H-MARL with an immutable predictor lock."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.portfolio_decision import compare_portfolio_decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-lock", required=True, type=Path)
    parser.add_argument("--prediction-bank", type=Path, default=None, help="Optional relocated bank; hashes must match lock.")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/test/signal_allocation_neural.yaml")
    parser.add_argument("--decision-config", type=Path, default=ROOT / "configs/research/portfolio_decision.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/core/env.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--development-only", action="store_true", help="Train/probe/inner only; skip outer evaluation and statistical gates.")
    parser.add_argument("--fold", default=None, help="Optional locked fold, e.g. development_2021.")
    args = parser.parse_args()
    _, output = compare_portfolio_decisions(args.predictor_lock, args.allocation_config, args.decision_config,
        args.raw_dir, args.assets_config, args.env_config, args.output_dir, args.prediction_bank,
        development_only=args.development_only, fold_id=args.fold)
    print(f"Comparison artifacts: {output}")


if __name__ == "__main__":
    main()
