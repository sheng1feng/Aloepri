import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from src.model_loader import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-F full-layer evaluation.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_f/full_layers.json")
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--attention-profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args, remaining = parser.parse_known_args()
    _, model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    layer_count = model.config.num_hidden_layers
    sys.argv = [
        sys.argv[0],
        "--model-dir",
        args.model_dir,
        "--output-path",
        args.output_path,
        "--layer-count",
        str(layer_count),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--lambda",
        str(args.lam),
        "--h",
        str(args.h),
        "--attention-profile",
        args.attention_profile,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ] + remaining
    from scripts.run_stage_f_prefix_layers import main as run_main

    run_main()


if __name__ == "__main__":
    main()
