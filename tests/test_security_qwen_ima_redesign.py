from src.security_qwen.ima import load_ima_embedding_sources


def test_stage_j_redesign_ima_loader_reads_buffered_embedding() -> None:
    baseline_embed, observed_embed, aux = load_ima_embedding_sources(target_name="stage_j_redesign")
    assert baseline_embed.ndim == 2
    assert observed_embed.ndim == 2
    assert observed_embed.shape[0] == baseline_embed.shape[0]
    assert observed_embed.shape[1] >= baseline_embed.shape[1]
    assert aux["resolved_target"]["name"] == "stage_j_redesign"
