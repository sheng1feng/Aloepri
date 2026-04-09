from src.security_qwen import default_phase0_datasets, default_security_targets, security_matrix_payload


def test_security_targets_cover_stage_a_h_j_k() -> None:
    names = [item.name for item in default_security_targets()]
    assert "stage_a_standard" in names
    assert "stage_h_full_obfuscated" in names
    assert "stage_j_stable_reference" in names
    assert "stage_k_tiny_a" in names


def test_phase0_dataset_catalog_has_three_dataset_kinds() -> None:
    datasets = default_phase0_datasets()
    kinds = [item.kind for item in datasets]
    assert kinds == ["smoke", "inversion", "frequency"]


def test_security_matrix_includes_deployment_intermediate_attacks() -> None:
    payload = security_matrix_payload()
    assert payload["format"] == "qwen_security_matrix_v1"
    deployment_matrix = payload["matrices"]["deployment_intermediate_attacks"]
    assert deployment_matrix["attacks"] == ["isa"]
    assert "kv_cache" in deployment_matrix["observable_types"]
