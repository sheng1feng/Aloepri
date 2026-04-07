from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.hidden_keys import build_hidden_transform, generate_hidden_permutation, generate_hidden_scaling
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.obfuscate_ffn import build_ffn_transform, generate_ffn_permutation, generate_ffn_scaling
from src.obfuscate_rmsnorm import estimate_kappa
from src.stage_b import StageBHiddenPermutationModel, TraceRecorder, prepare_stage_a_model, run_stage_b_single_prompt
from src.stage_c import StageCConfig, attach_stage_c_hooks


def test_estimate_kappa_stable() -> None:
    transform = build_hidden_transform(
        generate_hidden_permutation(32, seed=7),
        generate_hidden_scaling(32, (0.95, 1.05), seed=11),
    )
    kappa = estimate_kappa(transform, hidden_size=32, num_samples=512, seed=13)
    assert 0.9 < kappa < 1.1


def test_stage_c_full_block_improves_norm_mlp_and_block() -> None:
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
    layer0 = baseline_model.model.layers[0]
    ffn_transform = build_ffn_transform(
        generate_ffn_permutation(layer0.mlp.gate_proj.out_features, seed=DEFAULT_SEED + 303),
        generate_ffn_scaling(layer0.mlp.gate_proj.out_features, (0.95, 1.05), seed=DEFAULT_SEED + 404),
    )
    kappa = estimate_kappa(hidden_transform, hidden_size=baseline_model.config.hidden_size, num_samples=512, seed=DEFAULT_SEED + 505)
    stage_c_config = StageCConfig(
        hidden_transform=hidden_transform,
        kappa_input=kappa,
        kappa_post_attn=kappa,
        ffn_transform=ffn_transform,
    )

    stage_a_hidden_only, perm_vocab, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_attn, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_full, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)

    baseline_recorder = TraceRecorder()
    hidden_only_recorder = TraceRecorder()
    attn_only_recorder = TraceRecorder()
    full_block_recorder = TraceRecorder()

    hidden_only_model = StageBHiddenPermutationModel(stage_a_hidden_only, hidden_transform, recorder=hidden_only_recorder)
    attn_only_model = StageBHiddenPermutationModel(stage_a_attn, hidden_transform, recorder=attn_only_recorder)
    full_block_model = StageBHiddenPermutationModel(stage_a_full, hidden_transform, recorder=full_block_recorder)

    baseline_cleanup = attach_stage_c_hooks(
        baseline_model,
        baseline_recorder,
        attention_mode="plain",
        capture_embed_output=True,
    )
    hidden_only_cleanup = attach_stage_c_hooks(
        hidden_only_model.stage_a_model,
        hidden_only_recorder,
        attention_mode="plain",
        stage_c_config=stage_c_config,
        input_norm_mode="plain",
        post_attn_norm_mode="plain",
        ffn_mode="plain",
    )
    attn_only_cleanup = attach_stage_c_hooks(
        attn_only_model.stage_a_model,
        attn_only_recorder,
        attention_mode="wrapper",
        stage_c_config=stage_c_config,
        input_norm_mode="plain",
        post_attn_norm_mode="plain",
        ffn_mode="plain",
    )
    full_block_cleanup = attach_stage_c_hooks(
        full_block_model.stage_a_model,
        full_block_recorder,
        attention_mode="wrapper",
        stage_c_config=stage_c_config,
        input_norm_mode="wrapper",
        post_attn_norm_mode="wrapper",
        ffn_mode="wrapper",
    )

    try:
        prompt = DEFAULT_PROMPTS[0]
        hidden_only_result = run_stage_b_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_b_model=hidden_only_model,
            perm_vocab=perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=hidden_only_recorder,
            hidden_transform=hidden_transform,
        )
        attn_only_result = run_stage_b_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_b_model=attn_only_model,
            perm_vocab=perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=attn_only_recorder,
            hidden_transform=hidden_transform,
        )
        full_block_result = run_stage_b_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_b_model=full_block_model,
            perm_vocab=perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=full_block_recorder,
            hidden_transform=hidden_transform,
        )
    finally:
        baseline_cleanup()
        hidden_only_cleanup()
        attn_only_cleanup()
        full_block_cleanup()

    assert float(full_block_result.metrics["layer_0_input_norm_out_restored_max_abs_error"]) < float(
        hidden_only_result.metrics["layer_0_input_norm_out_restored_max_abs_error"]
    )
    assert float(full_block_result.metrics["layer_0_mlp_out_restored_max_abs_error"]) < float(
        attn_only_result.metrics["layer_0_mlp_out_restored_max_abs_error"]
    )
    assert float(full_block_result.metrics["layer_0_block_out_restored_max_abs_error"]) < float(
        attn_only_result.metrics["layer_0_block_out_restored_max_abs_error"]
    )
