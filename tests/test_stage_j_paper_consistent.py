from src.stage_j_paper_consistent import build_stage_j_paper_consistent_target


def test_stage_j_paper_consistent_target_marks_bridge_as_non_final() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["bridge_is_final_target"] is False
    assert payload["standard_graph_required"] is True
