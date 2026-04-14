from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_materialize import build_stage_j_redesign_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the redesigned Stage-J artifact skeleton.")
    parser.add_argument("--artifact-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--output-path", default="outputs/stage_j/redesign_regression.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_stage_j_redesign_regression(args.artifact_dir)
    write_json(args.output_path, payload)
    print(f"Saved redesigned Stage-J regression summary to {args.output_path}")


if __name__ == "__main__":
    main()
