from src.aloepri.catalog import default_stage_catalog, stage_catalog_payload


def test_stage_catalog_covers_a_to_k() -> None:
    stages = [entry.stage for entry in default_stage_catalog()]
    assert stages == ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def test_stage_catalog_payload_has_modular_entrypoints() -> None:
    payload = stage_catalog_payload()
    assert payload["format"] == "aloepri_stage_catalog_v1"
    stage_i = next(item for item in payload["stages"] if item["stage"] == "I")
    assert any("pipelines/stage_a" in entry or "pipelines.stage_a" in entry for entry in stage_i["modular_entrypoints"])
