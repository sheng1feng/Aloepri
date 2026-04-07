import torch

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.hidden_keys import (
    build_hidden_transform,
    generate_hidden_permutation,
    generate_hidden_scaling,
)
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import (
    StageBHiddenPermutationModel,
    TraceRecorder,
    attach_stage_b_hooks,
    fuse_block0_attention_hidden_transform,
    prepare_stage_a_model,
    run_stage_b_single_prompt,
)


def test_stage_b_wrapper_improves_attention_alignment() -> None:
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

    stage_a_hidden_only, perm_vocab, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_wrapper, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_fused, _, _ = prepare_stage_a_model(baseline_model, tokenizer, DEFAULT_SEED)
    stage_a_fused = fuse_block0_attention_hidden_transform(stage_a_fused, hidden_transform)

    baseline_recorder = TraceRecorder()
    hidden_only_recorder = TraceRecorder()
    wrapper_recorder = TraceRecorder()
    fused_recorder = TraceRecorder()

    hidden_only_model = StageBHiddenPermutationModel(stage_a_hidden_only, hidden_transform, recorder=hidden_only_recorder)
    wrapper_model = StageBHiddenPermutationModel(stage_a_wrapper, hidden_transform, recorder=wrapper_recorder)
    fused_model = StageBHiddenPermutationModel(stage_a_fused, hidden_transform, recorder=fused_recorder)

    baseline_cleanup = attach_stage_b_hooks(
        baseline_model,
        baseline_recorder,
        attention_mode="plain",
        capture_embed_output=True,
    )
    hidden_only_cleanup = attach_stage_b_hooks(
        hidden_only_model.stage_a_model,
        hidden_only_recorder,
        attention_mode="plain",
        capture_embed_output=False,
    )
    wrapper_cleanup = attach_stage_b_hooks(
        wrapper_model.stage_a_model,
        wrapper_recorder,
        attention_mode="wrapper",
        hidden_transform=hidden_transform,
        capture_embed_output=False,
    )
    fused_cleanup = attach_stage_b_hooks(
        fused_model.stage_a_model,
        fused_recorder,
        attention_mode="plain",
        capture_embed_output=False,
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
        wrapper_result = run_stage_b_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_b_model=wrapper_model,
            perm_vocab=perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=wrapper_recorder,
            hidden_transform=hidden_transform,
        )
        fused_result = run_stage_b_single_prompt(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            stage_b_model=fused_model,
            perm_vocab=perm_vocab,
            baseline_recorder=baseline_recorder,
            observed_recorder=fused_recorder,
            hidden_transform=hidden_transform,
        )
    finally:
        baseline_cleanup()
        hidden_only_cleanup()
        wrapper_cleanup()
        fused_cleanup()

    hidden_only_error = float(hidden_only_result.metrics["layer_0_attn_out_restored_max_abs_error"])
    wrapper_error = float(wrapper_result.metrics["layer_0_attn_out_restored_max_abs_error"])
    fused_error = float(fused_result.metrics["layer_0_attn_out_restored_max_abs_error"])

    assert hidden_only_error > 1e-6
    assert wrapper_error < hidden_only_error
    assert abs(fused_error - wrapper_error) < 1e-4

