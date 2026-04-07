from src.stage_k_llama_release import default_stage_k_llama_profiles


def test_stage_k_llama_profiles_include_stable_and_tiny() -> None:
    profiles = default_stage_k_llama_profiles()
    names = [item.name for item in profiles]
    assert "stable_reference" in names
    assert "tiny_a" in names
