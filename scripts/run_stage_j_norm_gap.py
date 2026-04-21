from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_norm_gap import build_stage_j_norm_gap_report


def main() -> None:
    payload = build_stage_j_norm_gap_report()
    write_json("outputs/stage_j/norm_gap_report.json", payload)
    print("Saved Stage-J norm gap report to outputs/stage_j/norm_gap_report.json")


if __name__ == "__main__":
    main()
