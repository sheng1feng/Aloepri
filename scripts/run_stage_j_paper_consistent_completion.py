from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_paper_consistent import build_stage_j_paper_consistent_evidence_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-J paper-consistent completion evidence bundle.")
    parser.add_argument("--candidate-dir", default="artifacts/stage_j_qwen_paper_consistent")
    parser.add_argument("--source-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--output-dir", default="outputs/stage_j/paper_consistent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=args.candidate_dir,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    print(payload["completion_summary"]["completion_status"])


if __name__ == "__main__":
    main()
