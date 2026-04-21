from src.stage_j_component_gap import build_stage_j_component_gap_report


def test_component_gap_report_flags_norm_as_unresolved() -> None:
    payload = build_stage_j_component_gap_report()
    assert payload["norm"]["status"] == "partially_resolved"
