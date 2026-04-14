from src.security_qwen.vma import infer_vma_default_projection_layers, load_vma_weight_sources


def test_stage_j_redesign_vma_loader_reads_buffered_weights() -> None:
    _, observed_weights, baseline_projection_weights, observed_projection_weights, _ = load_vma_weight_sources(
        target_name="stage_j_redesign"
    )
    assert "embed" in observed_weights
    assert "head" in observed_weights
    assert observed_weights["embed"].ndim == 2
    assert len(baseline_projection_weights) > 0
    assert len(observed_projection_weights) > 0


def test_infer_vma_default_projection_layers_returns_three_layers_for_stage_j_redesign() -> None:
    layers = infer_vma_default_projection_layers(target_name="stage_j_redesign")
    assert len(layers) == 3
    assert layers[0] == 0
