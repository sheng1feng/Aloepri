from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security_qwen.gate5_scan import run_gate5_scan


def main() -> None:
    output_path = Path("outputs/security_qwen/summary/gate5_parameter_scan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = run_gate5_scan()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Gate-5 parameter scan to {output_path}")


if __name__ == "__main__":
    main()
