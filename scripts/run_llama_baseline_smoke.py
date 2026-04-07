import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.evaluator import write_json
from src.llama_local_dev import tokenize_llama_prompt
from src.model_loader import load_model_and_tokenizer, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline smoke test for a real Llama checkpoint.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/llama_baseline_smoke.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_dir, device=args.device, dtype=args.dtype)
    encoded = tokenize_llama_prompt(tokenizer, args.prompt, device=args.device)
    generated = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    prompt_len = encoded["input_ids"].shape[1]
    generated_ids = generated[0, prompt_len:].detach().cpu()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    payload = {
        "phase": "llama_baseline_smoke",
        "model_dir": args.model_dir,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "prompt": args.prompt,
        "generated_ids": generated_ids.tolist(),
        "generated_text": generated_text,
    }
    write_json(args.output_path, payload)
    print(generated_text)
    print(f"Saved baseline smoke report to {args.output_path}")


if __name__ == "__main__":
    main()
