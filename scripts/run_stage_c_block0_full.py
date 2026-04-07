import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
    DEFAULT_STAGE_B_DTYPE,
    DEFAULT_STAGE_B_SCALE_RANGE,
    DEFAULT_STAGE_C_FFN_SCALE_RANGE,
    DEFAULT_STAGE_C_KAPPA_SAMPLES,
)
from src.evaluator import write_json
from src.hidden_keys import (
    build_hidden_transform,
    generate_hidden_permutation,
    generate_hidden_scaling,
)
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.obfuscate_ffn import build_ffn_transform, generate_ffn_permutation, generate_ffn_scaling
from src.obfuscate_rmsnorm import estimate_kappa
from src.stage_b import (
    StageBHiddenPermutationModel,
    TraceRecorder,
    aggregate_stage_b_results,
    prepare_stage_a_model,
    run_stage_b_single_prompt,
)
from src.stage_c import StageCConfig, attach_stage_c_hooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-C full block0 recovery experiment.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_c/block0_full.json")
    parser.add_argument("--dtype", default=DEFAULT_STAGE_B_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hidden-scale-low", type=float, default=DEFAULT_STAGE_B_SCALE_RANGE[0])
    parser.add_argument("--hidden-scale-high", type=float, default=DEFAULT_STAGE_B_SCALE_RANGE[1])
    parser.add_argument("--ffn-scale-low", type=float, default=DEFAULT_STAGE_C_FFN_SCALE_RANGE[0])
    parser.add_argument("--ffn-scale-high", type=float, default=DEFAULT_STAGE_C_FFN_SCALE_RANGE[1])
    parser.add_argument("--kappa-samples", type=int, default=DEFAULT_STAGE_C_KAPPA_SAMPLES)
    return parser.parse_args()


def run_mode(
    *,
    baseline_model,
    tokenizer,
    prompts,
    stage_a_model,
    perm_vocab,
    hidden_transform,
    stage_c_config,
    attention_mode,
    input_norm_mode,
    post_attn_norm_mode,
    ffn_mode,
):
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    stage_model = StageBHiddenPermutationModel(
        stage_a_model=stage_a_model,
        hidden_transform=hidden_transform,
        recorder=observed_recorder,
    )

    baseline_cleanup = attach_stage_c_hooks(
        baseline_model,
        baseline_recorder,
        attention_mode="plain",
        stage_c_config=None,
        input_norm_mode="plain",
        post_attn_norm_mode="plain",
        ffn_mode="plain",
        capture_embed_output=True,
    )
    observed_cleanup = attach_stage_c_hooks(
        stage_model.stage_a_model,
        observed_recorder,
        attention_mode=attention_mode,
        stage_c_config=stage_c_config,
        input_norm_mode=input_norm_mode,
        post_attn_norm_mode=post_attn_norm_mode,
        ffn_mode=ffn_mode,
        capture_embed_output=False,
    )

    try:
        results = [
            run_stage_b_single_prompt(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                prompt=prompt,
                stage_b_model=stage_model,
                perm_vocab=perm_vocab,
                baseline_recorder=baseline_recorder,
                observed_recorder=observed_recorder,
                hidden_transform=hidden_transform,
            )
            for prompt in prompts
        ]
    finally:
        baseline_cleanup()
        observed_cleanup()

    return {
        "summary": aggregate_stage_b_results(results),
        "prompts": [
            {
                "prompt": result.prompt,
                "mapped_input_ids": result.mapped_input_ids,
                **result.metrics,
            }
            for result in results
        ],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(
        model_name=args.model_dir,
        device="cpu",
        dtype=args.dtype,
    )

    hidden_transform = build_hidden_transform(
        generate_hidden_permutation(baseline_model.config.hidden_size, seed=args.seed + 101),
        generate_hidden_scaling(
            baseline_model.config.hidden_size,
            scale_range=(args.hidden_scale_low, args.hidden_scale_high),
            seed=args.seed + 202,
        ),
    )

    layer0 = baseline_model.model.layers[0]
    intermediate_size = layer0.mlp.gate_proj.out_features
    ffn_transform = build_ffn_transform(
        generate_ffn_permutation(intermediate_size, seed=args.seed + 303),
        generate_ffn_scaling(
            intermediate_size,
            scale_range=(args.ffn_scale_low, args.ffn_scale_high),
            seed=args.seed + 404,
        ),
    )

    kappa = estimate_kappa(
        hidden_transform=hidden_transform,
        hidden_size=baseline_model.config.hidden_size,
        num_samples=args.kappa_samples,
        seed=args.seed + 505,
    )
    stage_c_config = StageCConfig(
        hidden_transform=hidden_transform,
        kappa_input=kappa,
        kappa_post_attn=kappa,
        ffn_transform=ffn_transform,
    )

    stage_a_hidden_only, perm_vocab, _ = prepare_stage_a_model(baseline_model, tokenizer, args.seed)
    stage_a_attn, _, _ = prepare_stage_a_model(baseline_model, tokenizer, args.seed)
    stage_a_full, _, _ = prepare_stage_a_model(baseline_model, tokenizer, args.seed)

    hidden_only = run_mode(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        stage_a_model=stage_a_hidden_only,
        perm_vocab=perm_vocab,
        hidden_transform=hidden_transform,
        stage_c_config=stage_c_config,
        attention_mode="plain",
        input_norm_mode="plain",
        post_attn_norm_mode="plain",
        ffn_mode="plain",
    )
    attn_only = run_mode(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        stage_a_model=stage_a_attn,
        perm_vocab=perm_vocab,
        hidden_transform=hidden_transform,
        stage_c_config=stage_c_config,
        attention_mode="wrapper",
        input_norm_mode="plain",
        post_attn_norm_mode="plain",
        ffn_mode="plain",
    )
    full_block = run_mode(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        stage_a_model=stage_a_full,
        perm_vocab=perm_vocab,
        hidden_transform=hidden_transform,
        stage_c_config=stage_c_config,
        attention_mode="wrapper",
        input_norm_mode="wrapper",
        post_attn_norm_mode="wrapper",
        ffn_mode="wrapper",
    )

    tracked_metrics = [
        "avg_layer_0_attn_out_restored_max_abs_error",
        "avg_layer_0_mlp_out_restored_max_abs_error",
        "avg_layer_0_block_out_restored_max_abs_error",
        "avg_final_logits_restored_max_abs_error",
    ]
    comparison = {
        metric: {
            "hidden_only": hidden_only["summary"].get(metric),
            "attn_only": attn_only["summary"].get(metric),
            "full_block": full_block["summary"].get(metric),
        }
        for metric in tracked_metrics
    }

    payload = {
        "mode": "stage_c_block0_full",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "hidden_scale_range": [args.hidden_scale_low, args.hidden_scale_high],
        "ffn_scale_range": [args.ffn_scale_low, args.ffn_scale_high],
        "kappa_samples": args.kappa_samples,
        "kappa": kappa,
        "hidden_only": hidden_only,
        "attn_only": attn_only,
        "full_block": full_block,
        "comparison": comparison,
    }
    write_json(args.output_path, payload)
    print(f"Saved stage-C block0 full report to {args.output_path}")


if __name__ == "__main__":
    main()
