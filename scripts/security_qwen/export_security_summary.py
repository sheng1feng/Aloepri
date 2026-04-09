from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security_qwen.summary import security_summary_payload


def main() -> None:
    output_path = Path("outputs/security_qwen/summary/security_catalog.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(security_summary_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved security summary to {output_path}")


if __name__ == "__main__":
    main()
