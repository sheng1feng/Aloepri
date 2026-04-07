from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.hidden_keys import build_hidden_transform, generate_hidden_permutation, generate_hidden_scaling
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import StageBHiddenPermutationModel, prepare_stage_a_model, TraceRecorder
from src.stage_d import build_layer_configs, calibrate_layer_kappas, run_stage_d_single_prompt
from src.stage_e import attach_stage_e_hooks, build_layer_stage_e_configs


def test_stage_e_block0_profile_stable() -> None:
    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(
        DEFAULT_MODEL_DIR,
        device="cpu",
        dtype="float32",
    )
    hidden_transform = build_hidden_transform(
        generate_hidden_permutation(baseline_model.config.hidden_size, seed=DEFAULT_SEED + 101),
        generate_hidden_scaling(baseline_model.config.hidden_size, (0.95, 1.05), seed=DEFAULT_SEED + 202),
    )
    kappas = calibrate_layer_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:2],
        hidden_transform=hidden_transform,
        trace_layers=[0],
    )
    ffn_configs = build_layer_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        trace_layers=[0],
        seed=DEFAULT_SEED,
        ffn_scale_range=(0.95, 1.05),
    )
    layer_configs = build_layer_stage_e_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        layer_indices=[0],
        ffn_configs=ffn_configs,
        attention_profile="rqk_hqk_block_taukv_taugroup",
        seed=DEFAULT_SEED,
        qk_scale_range=(0.95, 1.05),
        beta=2,
    )

    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_hidden_only, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    observed_recorder = TraceRecorder()
    hidden_only_recorder = TraceRecorder()
    baseline_recorder = TraceRecorder()
    stage_model = StageBHiddenPermutationModel(stage_a_model, hidden_transform, recorder=observed_recorder)
    hidden_only_model = StageBHiddenPermutationModel(stage_a_hidden_only, hidden_transform, recorder=hidden_only_recorder)

    baseline_cleanup = attach_stage_e_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[0],
        layer_configs={},
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=True,
    )
    observed_cleanup = attach_stage_e_hooks(
        stage_model.stage_a_model,
        observed_recorder,
        trace_layers=[0],
        layer_configs=layer_configs,
        adapted_attention_layers=[0],
        adapted_norm_layers=[0],
        adapted_ffn_layers=[0],
        capture_embed_output=False,
    )
    hidden_only_cleanup = attach_stage_e_hooks(
        hidden_only_model.stage_a_model,
        hidden_only_recorder,
        trace_layers=[0],
        layer_configs=layer_configs,
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=False,
    )
    try:
        hidden_only_result = run_stage_d_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=hidden_only_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=hidden_only_recorder,
            hidden_transform=hidden_transform,
            trace_layers=[0],
            max_new_tokens=2,
        )
        result = run_stage_d_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=stage_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=observed_recorder,
            hidden_transform=hidden_transform,
            trace_layers=[0],
            max_new_tokens=2,
        )
    finally:
        baseline_cleanup()
        observed_cleanup()
        hidden_only_cleanup()

    assert float(result.metrics["layer_0_attn_out_restored_max_abs_error"]) < float(
        hidden_only_result.metrics["layer_0_attn_out_restored_max_abs_error"]
    )
    assert float(result.metrics["layer_0_block_out_restored_max_abs_error"]) < float(
        hidden_only_result.metrics["layer_0_block_out_restored_max_abs_error"]
    )


def test_stage_e_tau_variants_change_raw_head_traces() -> None:
    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(
        DEFAULT_MODEL_DIR,
        device="cpu",
        dtype="float32",
    )
    hidden_transform = build_hidden_transform(
        generate_hidden_permutation(baseline_model.config.hidden_size, seed=DEFAULT_SEED + 101),
        generate_hidden_scaling(baseline_model.config.hidden_size, (0.95, 1.05), seed=DEFAULT_SEED + 202),
    )
    kappas = calibrate_layer_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=[DEFAULT_PROMPTS[0]],
        hidden_transform=hidden_transform,
        trace_layers=[0],
    )
    ffn_configs = build_layer_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        trace_layers=[0],
        seed=DEFAULT_SEED,
        ffn_scale_range=(0.95, 1.05),
    )
    configs_block = build_layer_stage_e_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        layer_indices=[0],
        ffn_configs=ffn_configs,
        attention_profile="rqk_hqk_block",
        seed=DEFAULT_SEED,
        qk_scale_range=(0.95, 1.05),
        beta=4,
    )
    configs_tau = build_layer_stage_e_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        layer_indices=[0],
        ffn_configs=ffn_configs,
        attention_profile="rqk_hqk_block_taukv_taugroup",
        seed=DEFAULT_SEED,
        qk_scale_range=(0.95, 1.05),
        beta=4,
    )

    stage_a_block, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_tau, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    block_recorder = TraceRecorder()
    tau_recorder = TraceRecorder()
    baseline_recorder = TraceRecorder()
    block_model = StageBHiddenPermutationModel(stage_a_block, hidden_transform, recorder=block_recorder)
    tau_model = StageBHiddenPermutationModel(stage_a_tau, hidden_transform, recorder=tau_recorder)

    baseline_cleanup = attach_stage_e_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[0],
        layer_configs={},
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=True,
    )
    block_cleanup = attach_stage_e_hooks(
        block_model.stage_a_model,
        block_recorder,
        trace_layers=[0],
        layer_configs=configs_block,
        adapted_attention_layers=[0],
        adapted_norm_layers=[0],
        adapted_ffn_layers=[0],
        capture_embed_output=False,
    )
    tau_cleanup = attach_stage_e_hooks(
        tau_model.stage_a_model,
        tau_recorder,
        trace_layers=[0],
        layer_configs=configs_tau,
        adapted_attention_layers=[0],
        adapted_norm_layers=[0],
        adapted_ffn_layers=[0],
        capture_embed_output=False,
    )

    try:
        run_stage_d_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=block_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=block_recorder,
            hidden_transform=hidden_transform,
            trace_layers=[0],
            max_new_tokens=1,
        )
        run_stage_d_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=DEFAULT_PROMPTS[0],
            stage_model=tau_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=tau_recorder,
            hidden_transform=hidden_transform,
            trace_layers=[0],
            max_new_tokens=1,
        )
    finally:
        baseline_cleanup()
        block_cleanup()
        tau_cleanup()

    diff = (block_recorder.tensors['layer_0_q_heads_post_inter_raw'] - tau_recorder.tensors['layer_0_q_heads_post_inter_raw']).abs().max().item()
    assert diff > 1e-5
