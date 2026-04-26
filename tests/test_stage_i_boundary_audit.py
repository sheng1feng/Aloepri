from src.stage_i_vllm import build_stage_i_boundary_audit


def test_stage_i_boundary_audit_tracks_standard_runtime_contract() -> None:
    payload = build_stage_i_boundary_audit()
    assert payload["runtime_graph_is_standard"] is True
    assert payload["custom_online_operator_required"] is False


def test_stage_i_boundary_audit_marks_paper_consistent_expression_proof() -> None:
    payload = build_stage_i_boundary_audit()
    assert payload["exported_artifact_proves_attention_expression"] is True
    assert payload["exported_artifact_proves_ffn_expression"] is True
    assert payload["exported_artifact_proves_norm_expression"] is True
    assert payload["release_catalog_carries_paper_consistent_lineage"] is True
