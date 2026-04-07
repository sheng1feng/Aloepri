from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage_g_regression import main as run_main


def main() -> None:
    sys.argv = [
        sys.argv[0],
        "--output-path",
        "outputs/stage_g/ffn_block0.json",
        "--layer-count",
        "1",
        "--mode",
        "ffn_fused",
    ] + sys.argv[1:]
    run_main()


if __name__ == "__main__":
    main()
