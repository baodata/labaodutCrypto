"""Compare GP/convex controls on development OOF/inner; outer requires opt-in."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threadpoolctl import threadpool_limits
from src.evaluation.gp_research import compare_gp_allocators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor-lock', type=Path, required=True)
    parser.add_argument('--gp-config', type=Path, default=ROOT/'configs/gp_allocator.yaml')
    parser.add_argument('--allocation-config', type=Path, default=ROOT/'configs/signal_allocation_neural.yaml')
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--assets-config', type=Path, default=ROOT/'configs/assets.yaml')
    parser.add_argument('--env-config', type=Path, default=ROOT/'configs/env.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Restart unfinished cases from cash in the same immutable configuration.')
    parser.add_argument('--fold', help='Optional single fold, development OOF/inner only.')
    parser.add_argument('--evaluate-development-outer', action='store_true',
        help='Explicit full-fold exploratory outer backtest; never 2024-2025 test.')
    args = parser.parse_args()
    if args.resume and args.output_dir:
        parser.error('Use --resume or --output-dir, not both.')
    with threadpool_limits(limits=1):
        compare_gp_allocators(args)


if __name__ == '__main__':
    main()
