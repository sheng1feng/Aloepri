from src.stage_k_release import default_stage_k_profiles


def test_stage_k_profiles_prefer_redesigned_stage_j_sources() -> None:
    profiles = default_stage_k_profiles()
    assert all("stage_j_qwen_redesign" in profile.source_dir for profile in profiles)
