import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri.catalog import stage_catalog_payload


def main() -> None:
    output_path = Path("outputs/aloepri_stage_catalog.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stage_catalog_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved AloePri stage catalog to {output_path}")


if __name__ == "__main__":
    main()
