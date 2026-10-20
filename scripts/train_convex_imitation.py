"""Fit a deterministic PPO actor to convex targets on train OOF only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.convex_imitation import run_convex_imitation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-lock", required=True, type=Path)
    parser.add_argument("--imitation-config", type=Path, default=ROOT / "configs/research/convex_imitation.yaml")
    parser.add_argument("--allocation-config", type=Path, default=ROOT / "configs/test/signal_allocation_neural.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/core/assets.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs/core/env.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = run_convex_imitation(args.predictor_lock, args.allocation_config, args.imitation_config,
        args.raw_dir, args.assets_config, args.env_config, args.output_dir)
    print(f"Completed imitation: {output}")


if __name__ == "__main__":
    main()
