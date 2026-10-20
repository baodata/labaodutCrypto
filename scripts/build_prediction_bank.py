"""Build label-free Ridge forecasts and separate executable development labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.prediction_research import run_prediction_research


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-config", type=Path, default=ROOT / "configs/prediction_bank.yaml")
    parser.add_argument("--research-protocol", type=Path, default=ROOT / "configs/quant_research_protocol.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--assets-config", type=Path, default=ROOT / "configs/assets.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--development-only", action="store_true", help="Forecast train OOF/inner only; never forecast outer.")
    args = parser.parse_args(argv)
    run_prediction_research(args.research_protocol, args.prediction_config,
                            args.raw_dir, args.assets_config, args.output_dir, development_only=args.development_only)


if __name__ == "__main__":
    main()
