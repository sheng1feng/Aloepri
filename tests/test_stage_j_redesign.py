from pathlib import Path

from src.stage_j_materialize import build_stage_j_redesign_manifest, export_stage_j_redesign_checkpoint


def test_stage_j_redesign_manifest_uses_stage_h_and_i_lineage() -> None:
    payload = build_stage_j_redesign_manifest()
    assert payload["stage"] == "J"
    assert payload["source_stages"] == ["H", "I"]


def test_stage_j_redesign_manifest_distinguishes_legacy_stage_j() -> None:
    payload = build_stage_j_redesign_manifest()
    assert payload["legacy_reference"] == "artifacts/stage_j_full_square"


def test_export_stage_j_redesign_checkpoint_writes_manifest_and_links(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_h_pretrained"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "server" / "config.json").write_text("{}", encoding="utf-8")
    (source_dir / "server" / "obfuscation_config.json").write_text(
        '{"attention_profile":"rqk_hqk_block_taukv_taugroup","lambda":0.3,"h":128,"alpha_e":0.1,"alpha_h":0.05}',
        encoding="utf-8",
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_redesign"
    result = export_stage_j_redesign_checkpoint(export_dir, source_dir=source_dir, materialize=False)

    assert (export_dir / "manifest.json").exists()
    assert (export_dir / "server").exists()
    assert (export_dir / "client").exists()
    assert result["server_dir"].name == "server"
    manifest = Path(export_dir / "manifest.json").read_text(encoding="utf-8")
    assert "component_expression" in manifest
