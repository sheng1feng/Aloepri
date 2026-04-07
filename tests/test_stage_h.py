from src.stage_h_noise import NoiseCalibrationCase, rank_noise_cases
from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_d import attach_stage_d_hooks
from src.stage_f import (
    build_default_stage_f_keymat,
    build_layer_stage_f_configs,
    calibrate_keymat_kappas,
    run_stage_f_single_prompt,
)
from src.stage_g import build_layer_stage_g_configs, build_stage_g_model
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model


def test_rank_noise_cases_prefers_better_generation_then_error() -> None:
    items = [
        {
            "name": "worse",
            "summary": {
                "generated_text_exact_match_rate": 0.4,
                "generated_ids_exact_match_rate": 0.4,
                "greedy_first_token_match_rate": 0.8,
                "avg_final_logits_restored_max_abs_error": 1.0,
                "avg_layer_23_block_out_max_abs_error": 1.0,
            },
        },
        {
            "name": "better",
            "summary": {
                "generated_text_exact_match_rate": 0.8,
                "generated_ids_exact_match_rate": 0.8,
                "greedy_first_token_match_rate": 1.0,
                "avg_final_logits_restored_max_abs_error": 2.0,
                "avg_layer_23_block_out_max_abs_error": 2.0,
            },
        },
    ]
    ranked = rank_noise_cases(items)
    assert ranked[0]["name"] == "better"


def test_stage_h_static_attention_matches_stage_g_bridge_block0() -> None:
    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="float32")
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=0.3, h=128, seed=DEFAULT_SEED)
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:1],
        keymat_transform=keymat_transform,
        trace_layers=[0],
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=[0],
        kappa_by_layer=kappas,
        attention_profile="rqk_hqk_block_taukv_taugroup",
        seed=DEFAULT_SEED,
        alpha_e=0.1,
        alpha_h=0.05,
    )
    layer_configs_g = build_layer_stage_g_configs(layer_configs_f)
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    baseline_recorder = TraceRecorder()
    bridge_recorder = TraceRecorder()
    static_recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[0],
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        bridge_model, perm_vocab, inv_perm_vocab = build_stage_g_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=DEFAULT_SEED,
            recorder=bridge_recorder,
            layer_configs=layer_configs_g,
            adapted_layers=[0],
            mode="attention_fused",
            alpha_e=0.1,
            alpha_h=0.05,
            use_keymat_head=True,
            beta=8,
            gamma=1e3,
        )
        static_model, _, _ = build_stage_h_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=DEFAULT_SEED,
            recorder=static_recorder,
            layer_configs=layer_configs_h,
            adapted_layers=[0],
            alpha_e=0.1,
            alpha_h=0.05,
            use_keymat_head=True,
            beta=8,
            gamma=1e3,
        )
        bridge_result = run_stage_f_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=bridge_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=bridge_recorder,
            trace_layers=[0],
            max_new_tokens=2,
        )
        static_result = run_stage_f_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=static_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=static_recorder,
            trace_layers=[0],
            max_new_tokens=2,
        )
    finally:
        cleanup()
    assert abs(float(bridge_result.metrics["layer_0_attn_out_max_abs_error"]) - float(static_result.metrics["layer_0_attn_out_max_abs_error"])) < 1e-3
    assert abs(float(bridge_result.metrics["layer_0_block_out_max_abs_error"]) - float(static_result.metrics["layer_0_block_out_max_abs_error"])) < 1e-3
