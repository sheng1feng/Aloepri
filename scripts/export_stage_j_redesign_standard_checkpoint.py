from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Stage-J standard-visible bridge artifact.")
    parser.add_argument("--export-dir", default="artifacts/stage_j_qwen_redesign_standard")
    parser.add_argument("--source-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--norm-strategy", default="auto", choices=["auto", "ones", "metric_diag_sqrt", "kappa_fused"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_stage_j_redesign_standard_bridge(
        args.export_dir,
        source_dir=args.source_dir,
        materialize=args.materialize,
        norm_strategy=args.norm_strategy,
    )
    print(f"Exported Stage-J standard-visible bridge artifact to {result['export_dir']}")


if __name__ == "__main__":
    main()
