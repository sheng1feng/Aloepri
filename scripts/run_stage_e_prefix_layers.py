import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage_e_ablation import main as run_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-E prefix-layer experiment with one profile.")
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--output-path", default=None)
    args, remaining = parser.parse_known_args()
    output_path = args.output_path or f"outputs/stage_e/prefix_layers_{args.layer_count}.json"
    sys.argv = [
        sys.argv[0],
        "--layer-count",
        str(args.layer_count),
        "--profiles",
        args.profile,
        "--output-path",
        output_path,
    ] + remaining
    run_main()


if __name__ == "__main__":
    main()
