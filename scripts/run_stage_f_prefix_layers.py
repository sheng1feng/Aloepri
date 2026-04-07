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
    build_layer_stage_f_configs,
    build_stage_f_model,
    calibrate_keymat_kappas,
    run_stage_f_single_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-F prefix-layer evaluation.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--attention-profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    total_layers = baseline_model.config.num_hidden_layers
    if args.layer_count < 1 or args.layer_count > total_layers:
        raise ValueError(f"layer-count must be in [1, {total_layers}]")
    trace_layers = list(range(args.layer_count))
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=args.lam, h=args.h, seed=args.seed)
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:2],
        keymat_transform=keymat_transform,
        trace_layers=trace_layers,
    )
    layer_configs = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=trace_layers,
        kappa_by_layer=kappas,
        attention_profile=args.attention_profile,
        seed=args.seed,
    )

    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    baseline_cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=trace_layers,
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab = build_stage_f_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=args.seed,
            recorder=observed_recorder,
            layer_configs=layer_configs,
            adapted_layers=trace_layers,
            alpha_e=0.0,
            alpha_h=0.0,
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
                trace_layers=trace_layers,
                max_new_tokens=args.max_new_tokens,
            )
            for prompt in DEFAULT_PROMPTS
        ]
    finally:
        baseline_cleanup()

    payload = {
        "mode": "stage_f_prefix_layers",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "layer_count": args.layer_count,
        "lambda": args.lam,
        "h": args.h,
        "attention_profile": args.attention_profile,
        "trace_layers": trace_layers,
        "kappa_by_layer": kappas,
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
    print(f"Saved stage-F prefix report to {args.output_path}")


if __name__ == "__main__":
    main()
