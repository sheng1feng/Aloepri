from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.stage_k_correctness import run_stage_k_release_correctness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-K release-surface correctness for active Qwen profiles.")
    parser.add_argument("--release-dir", default="artifacts/stage_k_release")
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_DIR}/stage_k_release/correctness")
    parser.add_argument("--profiles", nargs="*", default=["default", "reference"])
    parser.add_argument("--buffered-source-dir")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage_k_release_correctness(
        release_dir=args.release_dir,
        output_dir=args.output_dir,
        profiles=tuple(args.profiles),
        buffered_source_dir=args.buffered_source_dir,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Saved Stage-K correctness summary to {Path(args.output_dir).parent / 'correctness_summary.json'}")
    print(f"Profiles: {summary['profiles']}")


if __name__ == "__main__":
    main()
