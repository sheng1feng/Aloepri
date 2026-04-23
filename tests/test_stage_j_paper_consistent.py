from src.stage_j_paper_consistent import build_stage_j_paper_consistent_target


def test_stage_j_paper_consistent_target_marks_bridge_as_non_final() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["bridge_is_final_target"] is False
    assert payload["standard_graph_required"] is True


def test_stage_j_paper_consistent_target_uses_canonical_candidate_paths() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["canonical_candidate_dir"] == "artifacts/stage_j_qwen_paper_consistent"
    assert payload["evidence_dir"] == "outputs/stage_j/paper_consistent"
    assert payload["historical_bridge_dir"] == "artifacts/stage_j_qwen_redesign_standard"
    assert payload["completion_statuses"] == ["export_visible_complete", "not_complete"]
