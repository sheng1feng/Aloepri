from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge


def test_export_stage_j_standard_bridge_writes_manifest_and_standard_proof(tmp_path: Path) -> None:
    source_dir = tmp_path / "legacy_stage_j"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 12),
            "lm_head.weight": torch.zeros(8, 12),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_redesign_standard"
    result = export_stage_j_redesign_standard_bridge(export_dir, source_dir=source_dir, materialize=False)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["server_dir"].name == "server"
    assert manifest["track"] == "standard_visible_bridge"
    assert manifest["standard_weight_proof"]["is_standard_weight_export"] is True
    assert manifest["equivalence_to_buffered_redesign"] is False
