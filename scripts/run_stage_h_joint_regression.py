import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage_h_attention_static import main as run_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-H joint regressions.", allow_abbrev=False)
    parser.add_argument("--layer-count", type=int, default=24)
    parser.add_argument("--output-path", default=None)
    args, remaining = parser.parse_known_args()
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = "outputs/stage_h/full_layers.json" if args.layer_count == 24 else f"outputs/stage_h/prefix_layers_{args.layer_count}.json"
    sys.argv = [
        sys.argv[0],
        "--output-path",
        output_path,
        "--layer-count",
        str(args.layer_count),
    ] + remaining
    run_main()


if __name__ == "__main__":
    main()
