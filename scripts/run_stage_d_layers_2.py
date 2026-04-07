import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage_d_layers import main as run_main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--layer-count",
        "2",
        "--output-path",
        "outputs/stage_d/layers_2.json",
    ] + sys.argv[1:]
    run_main()

