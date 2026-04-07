import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_OUTPUT_DIR
from src.evaluator import write_json
from src.stage_i_vllm import build_phase2_feasibility_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Stage-I Phase-2 feasibility summary.")
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_i/phase2_feasibility.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.output_path, build_phase2_feasibility_summary())
    print(f"Saved Stage-I Phase-2 feasibility summary to {args.output_path}")


if __name__ == "__main__":
    main()
