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
    calibrate_keymat_kappas,
    run_stage_f_single_prompt,
)
from src.stage_g import build_layer_stage_g_configs, build_stage_g_model
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-H staticized attention regression.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.3)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--alpha-e", type=float, default=0.25)
    parser.add_argument("--alpha-h", type=float, default=0.1)
    parser.add_argument("--attention-profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--beta", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1e3)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def run_case(*, baseline_model, tokenizer, trace_layers, builder, max_new_tokens):
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=trace_layers,
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab = builder(observed_recorder)
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
                max_new_tokens=max_new_tokens,
            )
            for prompt in DEFAULT_PROMPTS
        ]
    finally:
        cleanup()
    return {
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
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=trace_layers,
        kappa_by_layer=kappas,
        attention_profile=args.attention_profile,
        seed=args.seed,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
    )
    layer_configs_g = build_layer_stage_g_configs(layer_configs_f)
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    adapted_layers = trace_layers
    payload = {
        "mode": "stage_h_attention_static",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "layer_count": args.layer_count,
        "lambda": args.lam,
        "h": args.h,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "beta": args.beta,
        "gamma": args.gamma,
        "attention_profile": args.attention_profile,
        "trace_layers": trace_layers,
        "bridge_baseline": run_case(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            trace_layers=trace_layers,
            builder=lambda recorder: build_stage_g_model(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                keymat_transform=keymat_transform,
                seed=args.seed,
                recorder=recorder,
                layer_configs=layer_configs_g,
                adapted_layers=adapted_layers,
                mode="attention_fused",
                alpha_e=args.alpha_e,
                alpha_h=args.alpha_h,
                use_keymat_head=True,
                beta=args.beta,
                gamma=args.gamma,
            ),
            max_new_tokens=args.max_new_tokens,
        ),
        "staticized_candidate": run_case(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            trace_layers=trace_layers,
            builder=lambda recorder: build_stage_h_model(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                keymat_transform=keymat_transform,
                seed=args.seed,
                recorder=recorder,
                layer_configs=layer_configs_h,
                adapted_layers=adapted_layers,
                alpha_e=args.alpha_e,
                alpha_h=args.alpha_h,
                use_keymat_head=True,
                beta=args.beta,
                gamma=args.gamma,
            ),
            max_new_tokens=args.max_new_tokens,
        ),
    }
    write_json(args.output_path, payload)
    print(f"Saved stage-H attention static report to {args.output_path}")


if __name__ == "__main__":
    main()
