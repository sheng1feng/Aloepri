import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import write_json
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_f import aggregate_stage_f_results, run_stage_f_single_prompt
from src.stage_j_block0 import attach_stage_j_block0_hooks, build_stage_j_block0_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-J block0 square-transform prototype regression.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_j/block0_square.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    baseline_cleanup = attach_stage_j_block0_hooks(
        baseline_model,
        baseline_recorder,
        transform=None,
        trace_layers=[0],
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_block0_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            seed=args.seed,
            alpha_e=args.alpha_e,
            global_scale=args.global_scale,
            recorder=observed_recorder,
        )
        observed_cleanup = attach_stage_j_block0_hooks(
            stage_model,
            observed_recorder,
            transform,
            trace_layers=[0],
            capture_embed_output=True,
        )
        try:
            results = [
                run_stage_f_single_prompt(
                    baseline_model=baseline_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    stage_model=stage_model,
                    perm_vocab=perm_vocab,
                    inv_perm_vocab=inv_perm_vocab,
                    baseline_recorder=baseline_recorder,
                    observed_recorder=observed_recorder,
                    trace_layers=[0],
                    max_new_tokens=args.max_new_tokens,
                )
                for prompt in DEFAULT_PROMPTS
            ]
        finally:
            observed_cleanup()
    finally:
        baseline_cleanup()

    payload = {
        "stage": "J",
        "variant": "block0_square_hidden_transform",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "alpha_e": args.alpha_e,
        "global_scale": args.global_scale,
        "trace_layers": [0],
        "summary": aggregate_stage_f_results(results),
        "prompts": [
            {
                "prompt": result.prompt,
                "mapped_input_ids": result.mapped_input_ids,
                **result.metrics,
            }
            for result in results
        ],
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-J block0 square report to {args.output_path}")


if __name__ == "__main__":
    main()
