from src.stage_hk_audit import build_redesigned_expression_audit


def test_redesigned_expression_audit_detects_stage_h_bootstrap_expression() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_h_source"]["attention_profile_present"] is True
    assert payload["stage_h_source"]["keymat_parameters_present"] is True


def test_redesigned_expression_audit_marks_stage_j_as_paper_consistent_standard_export() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_j"]["candidate_dir"] == "artifacts/stage_j_qwen_paper_consistent"
    assert payload["stage_j"]["has_component_level_expression_manifest"] is True
    assert payload["stage_j"]["has_standard_weight_key_layout"] is True
    assert payload["stage_j"]["completion_status"] == "export_visible_complete"


def test_redesigned_expression_audit_marks_stage_k_as_paper_consistent_packaging() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_k"]["points_to_paper_consistent_stage_j"] is True
