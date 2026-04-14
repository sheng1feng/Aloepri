from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standard-weight proof report for Stage J export.")
    parser.add_argument("--server-dir", default="artifacts/stage_j_qwen_redesign/server")
    parser.add_argument("--output-path", default="outputs/stage_j/standard_weight_proof.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_stage_j_standard_weight_proof(args.server_dir)
    write_json(args.output_path, payload)
    print(f"Saved Stage-J standard-weight proof to {args.output_path}")


if __name__ == "__main__":
    main()
