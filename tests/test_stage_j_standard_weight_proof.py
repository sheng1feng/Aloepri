from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof
from src.stage_j_materialize import build_stage_j_redesign_regression, export_stage_j_redesign_checkpoint


def test_build_stage_j_standard_weight_proof_detects_buffered_layout(tmp_path: Path) -> None:
    server_dir = tmp_path / "server"
    server_dir.mkdir(parents=True)
    save_file(
        {
            "buffer::stage_a_model.model.embed_tokens.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.lm_head.weight": torch.zeros(8, 12),
        },
        str(server_dir / "model.safetensors"),
    )

    payload = build_stage_j_standard_weight_proof(server_dir)
    assert payload["is_standard_weight_export"] is False
    assert payload["layout"] == "buffered_stage_style"
    assert "model.embed_tokens.weight" in payload["missing_standard_keys"]


def test_build_stage_j_standard_weight_proof_detects_standard_layout(tmp_path: Path) -> None:
    server_dir = tmp_path / "server"
    server_dir.mkdir(parents=True)
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 12),
            "lm_head.weight": torch.zeros(8, 12),
        },
        str(server_dir / "model.safetensors"),
    )

    payload = build_stage_j_standard_weight_proof(server_dir)
    assert payload["is_standard_weight_export"] is True
    assert payload["layout"] == "standard_weight_visible"


def test_export_stage_j_redesign_checkpoint_writes_standard_weight_proof_summary(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_h_pretrained"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "server" / "config.json").write_text("{}", encoding="utf-8")
    (source_dir / "server" / "obfuscation_config.json").write_text(
        '{"attention_profile":"rqk_hqk_block_taukv_taugroup","lambda":0.3,"h":128,"alpha_e":0.1,"alpha_h":0.05}',
        encoding="utf-8",
    )
    save_file(
        {
            "buffer::stage_a_model.model.embed_tokens.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.lm_head.weight": torch.zeros(8, 12),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_redesign"
    export_stage_j_redesign_checkpoint(export_dir, source_dir=source_dir, materialize=False)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["standard_weight_proof"]["is_standard_weight_export"] is False
    assert manifest["standard_weight_proof"]["layout"] == "buffered_stage_style"


def test_stage_j_redesign_regression_includes_standard_weight_proof(tmp_path: Path) -> None:
    export_dir = tmp_path / "stage_j_qwen_redesign"
    export_dir.mkdir(parents=True)
    (export_dir / "server").mkdir()
    (export_dir / "client").mkdir()
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_stages": ["H", "I"],
                "standard_weight_proof": {
                    "is_standard_weight_export": False,
                    "layout": "buffered_stage_style",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = build_stage_j_redesign_regression(export_dir)
    assert payload["checks"]["is_standard_weight_export"] is False
    assert payload["checks"]["weight_layout"] == "buffered_stage_style"
