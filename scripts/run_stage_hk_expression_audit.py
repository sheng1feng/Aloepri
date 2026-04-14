from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_hk_audit import build_redesigned_expression_audit


def main() -> None:
    payload = build_redesigned_expression_audit()
    output_path = Path("outputs/stage_hk/redesign_expression_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved redesigned deployment expression audit to {output_path}")


if __name__ == "__main__":
    main()
