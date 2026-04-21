from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_bridge_regression import run_stage_j_bridge_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare buffered redesign Stage J against the standard-visible bridge.")
    parser.add_argument("--buffered-server-dir", default="artifacts/stage_j_qwen_redesign/server")
    parser.add_argument("--bridge-server-dir", default="artifacts/stage_j_qwen_redesign_standard/server")
    parser.add_argument("--output-path", default="outputs/stage_j/bridge_regression.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_stage_j_bridge_regression(
        buffered_server_dir=args.buffered_server_dir,
        bridge_server_dir=args.bridge_server_dir,
    )
    write_json(args.output_path, payload)
    print(f"Saved Stage-J bridge regression to {args.output_path}")


if __name__ == "__main__":
    main()
