import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
)
from src.evaluator import collect_prompt_outputs, write_json
from src.model_loader import load_model_and_tokenizer, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference for fixed prompts.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_DIR}/baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(
        model_name=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_id, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        result = collect_prompt_outputs(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        result["prompt_id"] = prompt_id
        result["seed"] = args.seed
        result["model_dir"] = args.model_dir
        write_json(output_dir / f"{prompt_id:02d}.json", result)
        print(f"Saved baseline prompt {prompt_id} to {output_dir / f'{prompt_id:02d}.json'}")


if __name__ == "__main__":
    main()
