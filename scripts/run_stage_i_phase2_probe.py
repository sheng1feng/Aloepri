import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import write_json
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_f import build_default_stage_f_keymat, build_layer_stage_f_configs, calibrate_keymat_kappas
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe how much of Stage-H can be materialized back into standard HF weights.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--alpha-e", type=float, default=0.1)
    parser.add_argument("--alpha-h", type=float, default=0.05)
    parser.add_argument("--beta", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1e3)
    parser.add_argument("--attention-profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_i/phase2_probe.json")
    return parser.parse_args()


def _shape_list(tensor) -> list[int]:
    return list(tensor.shape)


def _component_item(name: str, source_shape: list[int], target_shape: list[int], note: str) -> dict:
    return {
        "component": name,
        "source_shape": source_shape,
        "target_shape": target_shape,
        "direct_shape_match": source_shape == target_shape,
        "note": note,
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    adapted_layers = [0]
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=args.lam, h=args.h, seed=args.seed)
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:2],
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
        kappa_by_layer=kappas,
        attention_profile=args.attention_profile,
        seed=args.seed,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
    )
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    stage_h_model, _, _ = build_stage_h_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        keymat_transform=keymat_transform,
        seed=args.seed,
        recorder=TraceRecorder(),
        layer_configs=layer_configs_h,
        adapted_layers=adapted_layers,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        use_keymat_head=True,
        beta=args.beta,
        gamma=args.gamma,
    )

    baseline_layer = baseline_model.model.layers[0]
    stage_layer = stage_h_model.stage_a_model.model.layers[0]
    stage_embed = stage_h_model.stage_a_model.model.embed_tokens
    stage_head = stage_h_model.stage_a_model.lm_head

    materialization = [
        _component_item(
            "embedding_weight",
            _shape_list(stage_embed.weight),
            _shape_list(baseline_model.model.embed_tokens.weight),
            "Stage-H embed weight is already KeyMat-expanded; cannot be copied into original HF embedding tensor.",
        ),
        _component_item(
            "lm_head_weight",
            _shape_list(stage_head.weight),
            _shape_list(baseline_model.lm_head.weight),
            "Stage-H head weight follows expanded hidden basis; target HF lm_head expects original hidden size.",
        ),
        _component_item(
            "layer0_input_layernorm_metric_matrix",
            _shape_list(stage_layer.input_layernorm.metric_matrix),
            _shape_list(baseline_layer.input_layernorm.weight),
            "Fused norm is expressed as a quadratic form matrix, not a standard RMSNorm weight vector.",
        ),
        _component_item(
            "layer0_q_proj_weight",
            _shape_list(stage_layer.self_attn.q_weight),
            _shape_list(baseline_layer.self_attn.q_proj.weight),
            "Staticized attention q_proj still consumes expanded hidden states.",
        ),
        _component_item(
            "layer0_k_proj_weight",
            _shape_list(stage_layer.self_attn.k_weight),
            _shape_list(baseline_layer.self_attn.k_proj.weight),
            "Staticized attention k_proj still consumes expanded hidden states.",
        ),
        _component_item(
            "layer0_v_proj_weight",
            _shape_list(stage_layer.self_attn.v_weight),
            _shape_list(baseline_layer.self_attn.v_proj.weight),
            "Staticized attention v_proj still consumes expanded hidden states.",
        ),
        _component_item(
            "layer0_o_proj_weight",
            _shape_list(stage_layer.self_attn.o_weight),
            _shape_list(baseline_layer.self_attn.o_proj.weight),
            "Staticized attention o_proj emits expanded hidden states.",
        ),
        _component_item(
            "layer0_gate_proj_weight",
            _shape_list(stage_layer.mlp.gate_weight),
            _shape_list(baseline_layer.mlp.gate_proj.weight),
            "Fused FFN gate_proj expects expanded hidden input.",
        ),
        _component_item(
            "layer0_up_proj_weight",
            _shape_list(stage_layer.mlp.up_weight),
            _shape_list(baseline_layer.mlp.up_proj.weight),
            "Fused FFN up_proj expects expanded hidden input.",
        ),
        _component_item(
            "layer0_down_proj_weight",
            _shape_list(stage_layer.mlp.down_weight),
            _shape_list(baseline_layer.mlp.down_proj.weight),
            "Fused FFN down_proj emits expanded hidden output.",
        ),
    ]

    payload = {
        "stage": "I",
        "phase": "I-B-probe",
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "seed": args.seed,
        "lambda": args.lam,
        "h": args.h,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "beta": args.beta,
        "gamma": args.gamma,
        "attention_profile": args.attention_profile,
        "baseline_hidden_size": int(baseline_model.config.hidden_size),
        "keymat_hidden_size": int(keymat_transform.hidden_size),
        "keymat_expanded_size": int(keymat_transform.expanded_size),
        "all_components_directly_copyable": all(item["direct_shape_match"] for item in materialization),
        "materialization": materialization,
        "summary": {
            "copyable_component_count": sum(1 for item in materialization if item["direct_shape_match"]),
            "blocked_component_count": sum(1 for item in materialization if not item["direct_shape_match"]),
            "primary_blocker": "expanded KeyMat hidden size changes standard HF tensor shapes; RMSNorm also changes operator semantics.",
        },
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-I Phase-2 probe to {args.output_path}")


if __name__ == "__main__":
    main()
