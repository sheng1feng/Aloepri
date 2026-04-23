from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_paper_consistent import export_stage_j_paper_consistent_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Stage-J paper-consistent candidate checkpoint artifact.")
    parser.add_argument("--export-dir", default="artifacts/stage_j_qwen_paper_consistent")
    parser.add_argument("--source-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--norm-strategy", default="kappa_fused", choices=["ones", "metric_diag_sqrt", "kappa_fused"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_stage_j_paper_consistent_candidate(
        args.export_dir,
        source_dir=args.source_dir,
        materialize=args.materialize,
        norm_strategy=args.norm_strategy,
    )
    print(f"Exported Stage-J paper-consistent candidate artifact to {result['export_dir']}")


if __name__ == "__main__":
    main()
