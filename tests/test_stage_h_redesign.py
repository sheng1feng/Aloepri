from src.stage_h import build_stage_h_deployable_inventory


def test_stage_h_deployable_inventory_marks_attention_diversity() -> None:
    payload = build_stage_h_deployable_inventory()
    assert payload["stage"] == "H"
    assert payload["attention"]["preserve_block_permutation"] is True
    assert payload["attention"]["preserve_head_or_group_diversity"] is True


def test_stage_h_deployable_inventory_marks_norm_correction() -> None:
    payload = build_stage_h_deployable_inventory()
    assert payload["norm"]["preserve_kappa_correction"] is True


def test_stage_h_deployable_inventory_has_legacy_note() -> None:
    payload = build_stage_h_deployable_inventory()
    assert "legacy_stage_h_scope" in payload["migration"]
