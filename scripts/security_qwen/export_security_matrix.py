from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security_qwen import security_matrix_payload


def main() -> None:
    output_path = Path("outputs/security_qwen/summary/security_matrix.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(security_matrix_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved security matrix to {output_path}")


if __name__ == "__main__":
    main()
