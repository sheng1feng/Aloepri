from src.stage_i_vllm import build_stage_i_deployability_matrix


def test_stage_i_deployability_matrix_references_stage_h_inventory() -> None:
    payload = build_stage_i_deployability_matrix()
    assert payload["stage"] == "I"
    assert payload["source_stage"] == "H"


def test_stage_i_deployability_matrix_tracks_standard_runtime_boundary() -> None:
    payload = build_stage_i_deployability_matrix()
    assert payload["runtime_boundary"]["standard_transformer_graph"] is True
    assert "attention_diversity" in payload["validated_components"]
