from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_materialize import export_stage_j_redesign_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the redesigned Stage-J Qwen artifact.")
    parser.add_argument("--export-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--source-dir", default="artifacts/stage_h_pretrained")
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_stage_j_redesign_checkpoint(
        args.export_dir,
        source_dir=args.source_dir,
        materialize=args.materialize,
    )
    print(f"Exported redesigned Stage-J artifact to {result['export_dir']}")


if __name__ == "__main__":
    main()
