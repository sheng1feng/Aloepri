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
    DEFAULT_STAGE_B_DTYPE,
    DEFAULT_STAGE_B_SCALE_RANGE,
    DEFAULT_STAGE_C_FFN_SCALE_RANGE,
)
from src.evaluator import write_json
from src.hidden_keys import build_hidden_transform, generate_hidden_permutation, generate_hidden_scaling
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import StageBHiddenPermutationModel, prepare_stage_a_model, TraceRecorder
from src.stage_d import (
    LayerStageDConfig,
    aggregate_stage_d_results,
    attach_stage_d_hooks,
    build_layer_configs,
    calibrate_layer_kappas,
    run_stage_d_single_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-D multi-layer recovery experiment.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--dtype", default=DEFAULT_STAGE_B_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hidden-scale-low", type=float, default=DEFAULT_STAGE_B_SCALE_RANGE[0])
    parser.add_argument("--hidden-scale-high", type=float, default=DEFAULT_STAGE_B_SCALE_RANGE[1])
    parser.add_argument("--ffn-scale-low", type=float, default=DEFAULT_STAGE_C_FFN_SCALE_RANGE[0])
    parser.add_argument("--ffn-scale-high", type=float, default=DEFAULT_STAGE_C_FFN_SCALE_RANGE[1])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def run_mode(
    *,
    baseline_model,
    tokenizer,
    prompts,
    stage_a_model,
    perm_vocab,
    inv_perm_vocab,
    hidden_transform,
    trace_layers,
    layer_configs: dict[int, LayerStageDConfig],
    adapted_layers,
    max_new_tokens: int,
):
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    stage_model = StageBHiddenPermutationModel(
        stage_a_model=stage_a_model,
        hidden_transform=hidden_transform,
        recorder=observed_recorder,
    )

    baseline_cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=trace_layers,
        layer_configs={},
        attention_mode="plain",
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=True,
        record_norm_inputs=False,
    )
    observed_cleanup = attach_stage_d_hooks(
        stage_model.stage_a_model,
        observed_recorder,
        trace_layers=trace_layers,
        layer_configs=layer_configs,
        attention_mode="wrapper",
        adapted_attention_layers=adapted_layers,
        adapted_norm_layers=adapted_layers,
        adapted_ffn_layers=adapted_layers,
        capture_embed_output=False,
        record_norm_inputs=False,
    )

    try:
        results = [
            run_stage_d_single_prompt(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                prompt=prompt,
                stage_model=stage_model,
                perm_vocab=perm_vocab,
                inv_perm_vocab=inv_perm_vocab,
                baseline_recorder=baseline_recorder,
                observed_recorder=observed_recorder,
                hidden_transform=hidden_transform,
                trace_layers=trace_layers,
                max_new_tokens=max_new_tokens,
            )
            for prompt in prompts
        ]
    finally:
        baseline_cleanup()
        observed_cleanup()

    return {
        "adapted_layers": list(adapted_layers),
        "summary": aggregate_stage_d_results(results),
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
    tokenizer, baseline_model = load_model_and_tokenizer(
        model_name=args.model_dir,
        device="cpu",
        dtype=args.dtype,
    )

    total_layers = baseline_model.config.num_hidden_layers
    if args.layer_count < 1 or args.layer_count > total_layers:
        raise ValueError(f"layer-count must be in [1, {total_layers}], got {args.layer_count}")
    trace_layers = list(range(args.layer_count))

    hidden_transform = build_hidden_transform(
        generate_hidden_permutation(baseline_model.config.hidden_size, seed=args.seed + 101),
        generate_hidden_scaling(
            baseline_model.config.hidden_size,
            scale_range=(args.hidden_scale_low, args.hidden_scale_high),
            seed=args.seed + 202,
        ),
    )

    kappa_by_layer = calibrate_layer_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        hidden_transform=hidden_transform,
        trace_layers=trace_layers,
    )
    layer_configs = build_layer_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappa_by_layer,
        trace_layers=trace_layers,
        seed=args.seed,
        ffn_scale_range=(args.ffn_scale_low, args.ffn_scale_high),
    )

    stage_a_hidden_only, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, args.seed)
    stage_a_block0, _, _ = prepare_stage_a_model(baseline_model, tokenizer, args.seed)
    stage_a_prefix, _, _ = prepare_stage_a_model(baseline_model, tokenizer, args.seed)

    hidden_only = run_mode(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        stage_a_model=stage_a_hidden_only,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        hidden_transform=hidden_transform,
        trace_layers=trace_layers,
        layer_configs=layer_configs,
        adapted_layers=[],
        max_new_tokens=args.max_new_tokens,
    )

    block0_only_layers = [0]
    block0_only = run_mode(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        stage_a_model=stage_a_block0,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        hidden_transform=hidden_transform,
        trace_layers=trace_layers,
        layer_configs=layer_configs,
        adapted_layers=block0_only_layers,
        max_new_tokens=args.max_new_tokens,
    )

    prefix_layers = list(range(args.layer_count))
    prefix_full = run_mode(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        stage_a_model=stage_a_prefix,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        hidden_transform=hidden_transform,
        trace_layers=trace_layers,
        layer_configs=layer_configs,
        adapted_layers=prefix_layers,
        max_new_tokens=args.max_new_tokens,
    )

    payload = {
        "mode": "stage_d_layers",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "layer_count": args.layer_count,
        "trace_layers": trace_layers,
        "hidden_scale_range": [args.hidden_scale_low, args.hidden_scale_high],
        "ffn_scale_range": [args.ffn_scale_low, args.ffn_scale_high],
        "kappa_by_layer": kappa_by_layer,
        "hidden_only": hidden_only,
        "block0_only": block0_only,
        "prefix_full": prefix_full,
    }
    write_json(args.output_path, payload)
    print(f"Saved stage-D report to {args.output_path}")


if __name__ == "__main__":
    main()

