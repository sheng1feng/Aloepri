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
from src.stage_d import attach_stage_d_hooks
from src.stage_f import (
    aggregate_stage_f_results,
    build_default_stage_f_keymat,
    build_stage_f_model,
    run_stage_f_single_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-F embed/head-only evaluation.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_f/embed_head_eval.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def run_case(
    *,
    baseline_model,
    tokenizer,
    keymat_transform,
    seed: int,
    alpha_e: float,
    alpha_h: float,
    max_new_tokens: int,
):
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    baseline_cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[],
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab = build_stage_f_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=seed,
            recorder=observed_recorder,
            layer_configs={},
            adapted_layers=[],
            alpha_e=alpha_e,
            alpha_h=alpha_h,
            use_keymat_head=True,
        )
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
                trace_layers=[],
                max_new_tokens=max_new_tokens,
            )
            for prompt in DEFAULT_PROMPTS
        ]
    finally:
        baseline_cleanup()
    return {
        "alpha_e": alpha_e,
        "alpha_h": alpha_h,
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


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=args.lam, h=args.h, seed=args.seed)
    payload = {
        "mode": "stage_f_embed_head",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "lambda": args.lam,
        "h": args.h,
        "cases": {
            "zero_noise": run_case(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                keymat_transform=keymat_transform,
                seed=args.seed,
                alpha_e=0.0,
                alpha_h=0.0,
                max_new_tokens=args.max_new_tokens,
            ),
            "small_noise": run_case(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                keymat_transform=keymat_transform,
                seed=args.seed + 100,
                alpha_e=0.01,
                alpha_h=0.01,
                max_new_tokens=args.max_new_tokens,
            ),
            "medium_noise": run_case(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                keymat_transform=keymat_transform,
                seed=args.seed + 200,
                alpha_e=0.05,
                alpha_h=0.05,
                max_new_tokens=args.max_new_tokens,
            ),
        },
    }
    write_json(args.output_path, payload)
    print(f"Saved stage-F embed/head report to {args.output_path}")


if __name__ == "__main__":
    main()
