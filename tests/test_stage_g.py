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


def test_stage_g_norm_fused_block0_matches_baseline() -> None:
    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="float32")
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=0.1, h=32, seed=DEFAULT_SEED)
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
    )
    layer_configs_g = build_layer_stage_g_configs(layer_configs_f)
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[0],
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab = build_stage_g_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=DEFAULT_SEED,
            recorder=observed_recorder,
            layer_configs=layer_configs_g,
            adapted_layers=[0],
            mode="norm_fused",
            alpha_e=0.0,
            alpha_h=0.0,
            use_keymat_head=True,
        )
        result = run_stage_f_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=stage_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=observed_recorder,
            trace_layers=[0],
            max_new_tokens=2,
        )
    finally:
        cleanup()
    assert float(result.metrics["layer_0_input_norm_out_max_abs_error"]) < 1e-3
    assert float(result.metrics["layer_0_block_out_max_abs_error"]) < 5e-4
    assert result.metrics["generated_ids_exact_match"] is True


def test_stage_g_attention_fused_prefix2_stays_stable() -> None:
    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="float32")
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=0.1, h=32, seed=DEFAULT_SEED)
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:1],
        keymat_transform=keymat_transform,
        trace_layers=[0, 1],
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=[0, 1],
        kappa_by_layer=kappas,
        attention_profile="rqk_hqk_block_taukv_taugroup",
        seed=DEFAULT_SEED,
    )
    layer_configs_g = build_layer_stage_g_configs(layer_configs_f)
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[0, 1],
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab = build_stage_g_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=DEFAULT_SEED,
            recorder=observed_recorder,
            layer_configs=layer_configs_g,
            adapted_layers=[0, 1],
            mode="attention_fused",
            alpha_e=0.0,
            alpha_h=0.0,
            use_keymat_head=True,
        )
        result = run_stage_f_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=stage_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=observed_recorder,
            trace_layers=[0, 1],
            max_new_tokens=2,
        )
    finally:
        cleanup()
    assert float(result.metrics["layer_1_block_out_max_abs_error"]) < 1e-3
    assert float(result.metrics["final_logits_restored_max_abs_error"]) < 2e-3
    assert result.metrics["generated_ids_exact_match"] is True
