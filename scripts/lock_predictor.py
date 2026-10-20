"""Lock the best inner-IC predictor per fold at a predeclared horizon."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.predictor_lock import make_predictor_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", default="ridge_fixed,momentum,mean_reversion,gru,gnn_gru")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    lock = make_predictor_lock(args.prediction_bank, args.output,
        candidates=args.candidates.split(","), horizon=args.horizon)
    for row in lock["selections"]:
        print(f"{row['fold_id']}: {row['family']} h={row['horizon_sessions']} "
              f"inner IC={row['selected_mean_rank_ic']:.4f}, cutoff={row['selection_cutoff']}")
    print(f"Locked: {args.output} / {lock['lock_digest']}")


if __name__ == "__main__":
    main()
