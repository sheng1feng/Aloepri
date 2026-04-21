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


def test_export_stage_j_standard_bridge_materializes_buffered_redesign_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "buffered_stage_j"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "server" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "buffer::stage_a_model.model.embed_tokens.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.lm_head.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.input_layernorm.metric_matrix": torch.diag(torch.tensor([4.0] * 12)),
            "buffer::stage_a_model.model.layers.0.post_attention_layernorm.metric_matrix": torch.diag(torch.tensor([9.0] * 12)),
            "buffer::stage_a_model.model.norm.metric_matrix": torch.diag(torch.tensor([16.0] * 12)),
            "buffer::stage_a_model.model.layers.0.self_attn.q_weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.k_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.v_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.o_weight": torch.zeros(12, 8),
            "buffer::stage_a_model.model.layers.0.self_attn.q_bias": torch.zeros(8),
            "buffer::stage_a_model.model.layers.0.self_attn.k_bias": torch.zeros(4),
            "buffer::stage_a_model.model.layers.0.self_attn.v_bias": torch.zeros(4),
            "buffer::stage_a_model.model.layers.0.mlp.gate_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.up_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.down_weight": torch.zeros(12, 16),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_redesign_standard"
    export_stage_j_redesign_standard_bridge(
        export_dir,
        source_dir=source_dir,
        materialize=False,
        norm_strategy="metric_diag_sqrt",
    )

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_source_dir"] == str(source_dir)
    assert manifest["standard_weight_proof"]["is_standard_weight_export"] is True
    assert manifest["standard_weight_proof"]["layout"] == "standard_weight_visible"
    assert manifest["bridge_source_layout"] == "buffered_stage_style"

    from safetensors.torch import load_file

    bridge_state = load_file(str(export_dir / "server" / "model.safetensors"))
    assert torch.allclose(bridge_state["model.layers.0.input_layernorm.weight"], torch.full((12,), 2.0))
    assert torch.allclose(bridge_state["model.layers.0.post_attention_layernorm.weight"], torch.full((12,), 3.0))
    assert torch.allclose(bridge_state["model.norm.weight"], torch.full((12,), 4.0))


def test_export_stage_j_standard_bridge_defaults_norms_to_ones(tmp_path: Path) -> None:
    source_dir = tmp_path / "buffered_stage_j"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "server" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "buffer::stage_a_model.model.embed_tokens.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.lm_head.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.input_layernorm.metric_matrix": torch.diag(torch.tensor([4.0] * 12)),
            "buffer::stage_a_model.model.norm.metric_matrix": torch.diag(torch.tensor([16.0] * 12)),
            "buffer::stage_a_model.model.layers.0.self_attn.q_weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.k_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.v_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.o_weight": torch.zeros(12, 8),
            "buffer::stage_a_model.model.layers.0.mlp.gate_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.up_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.down_weight": torch.zeros(12, 16),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_redesign_standard"
    export_stage_j_redesign_standard_bridge(export_dir, source_dir=source_dir, materialize=False)

    from safetensors.torch import load_file

    bridge_state = load_file(str(export_dir / "server" / "model.safetensors"))
    assert torch.allclose(bridge_state["model.layers.0.input_layernorm.weight"], torch.ones(12))
    assert torch.allclose(bridge_state["model.norm.weight"], torch.ones(12))
