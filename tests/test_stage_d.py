from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.hidden_keys import build_hidden_transform, generate_hidden_permutation, generate_hidden_scaling
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import StageBHiddenPermutationModel, prepare_stage_a_model, TraceRecorder
from src.stage_d import (
    aggregate_stage_d_results,
    attach_stage_d_hooks,
    build_layer_configs,
    calibrate_layer_kappas,
    run_stage_d_single_prompt,
)


def test_calibrate_layer_kappas_returns_two_layers() -> None:
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
        trace_layers=[0, 1],
    )
    assert set(kappas.keys()) == {0, 1}
    for values in kappas.values():
        assert 0.9 < values["input"] < 1.1
        assert 0.9 < values["post_attn"] < 1.1


def test_two_layer_prefix_improves_layer1_block_out() -> None:
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
        trace_layers=[0, 1],
    )
    layer_configs = build_layer_configs(
        baseline_model=baseline_model,
        hidden_transform=hidden_transform,
        kappa_by_layer=kappas,
        trace_layers=[0, 1],
        seed=DEFAULT_SEED,
        ffn_scale_range=(0.95, 1.05),
    )

    stage_a_block0, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_prefix, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)

    baseline_recorder = TraceRecorder()
    block0_recorder = TraceRecorder()
    prefix_recorder = TraceRecorder()

    block0_model = StageBHiddenPermutationModel(stage_a_block0, hidden_transform, recorder=block0_recorder)
    prefix_model = StageBHiddenPermutationModel(stage_a_prefix, hidden_transform, recorder=prefix_recorder)

    baseline_cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=[0, 1],
        layer_configs={},
        attention_mode="plain",
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=True,
    )
    block0_cleanup = attach_stage_d_hooks(
        block0_model.stage_a_model,
        block0_recorder,
        trace_layers=[0, 1],
        layer_configs=layer_configs,
        attention_mode="wrapper",
        adapted_attention_layers=[0],
        adapted_norm_layers=[0],
        adapted_ffn_layers=[0],
    )
    prefix_cleanup = attach_stage_d_hooks(
        prefix_model.stage_a_model,
        prefix_recorder,
        trace_layers=[0, 1],
        layer_configs=layer_configs,
        attention_mode="wrapper",
        adapted_attention_layers=[0, 1],
        adapted_norm_layers=[0, 1],
        adapted_ffn_layers=[0, 1],
    )

    try:
        prompt = DEFAULT_PROMPTS[0]
        block0_result = run_stage_d_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_model=block0_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=block0_recorder,
            hidden_transform=hidden_transform,
            trace_layers=[0, 1],
            max_new_tokens=2,
        )
        prefix_result = run_stage_d_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_model=prefix_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=prefix_recorder,
            hidden_transform=hidden_transform,
            trace_layers=[0, 1],
            max_new_tokens=2,
        )
    finally:
        baseline_cleanup()
        block0_cleanup()
        prefix_cleanup()

    assert float(prefix_result.metrics["layer_1_block_out_restored_max_abs_error"]) < float(
        block0_result.metrics["layer_1_block_out_restored_max_abs_error"]
    )
    summary = aggregate_stage_d_results([block0_result, prefix_result])
    assert "avg_final_logits_restored_max_abs_error" in summary
