import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_k_llama_release import export_stage_k_llama_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-K Llama release catalog from validated Llama artifacts.")
    parser.add_argument("--export-dir", default="artifacts/stage_k_llama_release")
    parser.add_argument("--materialize", action="store_true", help="Copy profile directories instead of creating symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = export_stage_k_llama_release(args.export_dir, materialize=args.materialize)
    print(f"Saved Stage-K Llama release catalog to {Path(args.export_dir) / 'catalog.json'}")
    print(f"Profiles: {[item['name'] for item in catalog['profiles']]}")


if __name__ == "__main__":
    main()
