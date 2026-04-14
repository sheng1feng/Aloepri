from src.security_qwen.isa import load_isa_model_bundle


def test_stage_j_redesign_isa_bundle_uses_stage_h_style_model() -> None:
    bundle = load_isa_model_bundle(target_name="stage_j_redesign")
    assert hasattr(bundle["model"], "stage_a_model")
