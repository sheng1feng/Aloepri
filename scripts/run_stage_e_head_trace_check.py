import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED, DEFAULT_STAGE_B_DTYPE
from src.evaluator import write_json
from src.hidden_keys import build_hidden_transform, generate_hidden_permutation, generate_hidden_scaling
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import StageBHiddenPermutationModel, prepare_stage_a_model, TraceRecorder
from src.stage_d import build_layer_configs, calibrate_layer_kappas, run_stage_d_single_prompt
from src.stage_e import attach_stage_e_hooks, build_layer_stage_e_configs


RAW_TRACE_NAMES = [
    "q_heads_pre_inter_raw",
    "k_heads_pre_inter_raw",
    "v_heads_pre_inter_raw",
    "q_heads_post_inter_raw",
    "k_heads_post_inter_raw",
    "v_heads_post_inter_raw",
    "attn_heads_pre_inverse_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw head-level traces for tau_kv/tau_group.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_e/head_trace_check.json")
    parser.add_argument("--dtype", default=DEFAULT_STAGE_B_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt", default=DEFAULT_PROMPTS[0])
    parser.add_argument("--layer-idx", type=int, default=0)
    return parser.parse_args()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_profile(
    *,
    baseline_model,
    tokenizer,
    prompt,
    stage_a_model,
    perm_vocab,
    inv_perm_vocab,
    hidden_transform,
    layer_configs,
    adapted_layers,
    trace_layers,
):
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    stage_model = StageBHiddenPermutationModel(stage_a_model, hidden_transform, recorder=observed_recorder)

    baseline_cleanup = attach_stage_e_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=trace_layers,
        layer_configs={},
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=True,
    )
    observed_cleanup = attach_stage_e_hooks(
        stage_model.stage_a_model,
        observed_recorder,
        trace_layers=trace_layers,
        layer_configs=layer_configs,
        adapted_attention_layers=adapted_layers,
        adapted_norm_layers=adapted_layers,
        adapted_ffn_layers=adapted_layers,
        capture_embed_output=False,
    )

    try:
        result = run_stage_d_single_prompt(
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
            max_new_tokens=2,
        )
        traces = {name: observed_recorder.tensors.get(name) for name in observed_recorder.tensors}
    finally:
        baseline_cleanup()
        observed_cleanup()

    return result, traces


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
        generate_hidden_scaling(baseline_model.config.hidden_size, (0.95, 1.05), seed=args.seed + 202),
    )
    trace_layers = [args.layer_idx]
    kappas = calibrate_layer_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=[args.prompt],
        hidden_transform=hidden_transform,
        trace_layers=trace_layers,
    )
    ffn_configs = build_layer_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        trace_layers=trace_layers,
        seed=args.seed,
        ffn_scale_range=(0.95, 1.05),
    )

    profiles = [
        "rqk_hqk_block",
        "rqk_hqk_block_taukv",
        "rqk_hqk_block_taukv_taugroup",
    ]
    payload = {
        "mode": "stage_e_head_trace_check",
        "prompt": args.prompt,
        "layer_idx": args.layer_idx,
        "profiles": {},
    }

    captured = {}
    for profile in profiles:
        stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, args.seed)
        layer_configs = build_layer_stage_e_configs(
            baseline_model=baseline_model,
            hidden_transform=hidden_transform,
            kappa_by_layer=kappas,
            layer_indices=trace_layers,
            ffn_configs=ffn_configs,
            attention_profile=profile,
            seed=args.seed,
            qk_scale_range=(0.95, 1.05),
            beta=4,
        )
        result, traces = run_profile(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            stage_a_model=stage_a_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            hidden_transform=hidden_transform,
            layer_configs=layer_configs,
            adapted_layers=trace_layers,
            trace_layers=trace_layers,
        )
        payload["profiles"][profile] = {
            "metrics": result.metrics,
            "tau_kv": None if layer_configs[args.layer_idx].attention_config.inter_head.tau_kv is None else layer_configs[args.layer_idx].attention_config.inter_head.tau_kv.tolist(),
            "tau_group": None if layer_configs[args.layer_idx].attention_config.inter_head.tau_group is None else layer_configs[args.layer_idx].attention_config.inter_head.tau_group.tolist(),
        }
        captured[profile] = traces

    layer_prefix = f"layer_{args.layer_idx}"
    comparisons = {}
    pairs = [
        ("rqk_hqk_block", "rqk_hqk_block_taukv"),
        ("rqk_hqk_block_taukv", "rqk_hqk_block_taukv_taugroup"),
    ]
    for left, right in pairs:
        pair_key = f"{left}__vs__{right}"
        comparisons[pair_key] = {}
        for trace_name in RAW_TRACE_NAMES:
            name = f"{layer_prefix}_{trace_name}"
            comparisons[pair_key][trace_name] = max_abs_diff(captured[left][name], captured[right][name])
    payload["raw_trace_comparisons"] = comparisons

    write_json(args.output_path, payload)
    print(f"Saved stage-E head trace diagnostics to {args.output_path}")


if __name__ == "__main__":
    main()
