from src.security_qwen.artifacts import get_security_target, resolve_security_target


def test_redesign_stage_j_target_resolves() -> None:
    spec = get_security_target("stage_j_redesign")
    assert spec.artifact_dir == "artifacts/stage_j_qwen_redesign"


def test_redesign_stage_k_target_resolves() -> None:
    resolved = resolve_security_target("stage_k_redesign_tiny_a")
    assert resolved.resolved_root_dir.endswith("artifacts/stage_k_release/profiles/tiny_a")


def test_redesign_stage_j_standard_bridge_target_can_be_added_later() -> None:
    from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof

    payload = build_stage_j_standard_weight_proof("artifacts/stage_j_qwen_redesign_standard/server")
    assert payload["is_standard_weight_export"] is True
