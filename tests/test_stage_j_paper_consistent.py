from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.stage_j_paper_consistent import (
    _build_export_visible_component_metadata,
    build_stage_j_paper_consistent_completion_summary,
    build_stage_j_paper_consistent_evidence_bundle,
    build_stage_j_paper_consistent_target,
    export_stage_j_paper_consistent_candidate,
)


def test_stage_j_paper_consistent_target_marks_bridge_as_non_final() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["bridge_is_final_target"] is False
    assert payload["standard_graph_required"] is True


def test_stage_j_paper_consistent_target_uses_canonical_candidate_paths() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["canonical_candidate_dir"] == "artifacts/stage_j_qwen_paper_consistent"
    assert payload["evidence_dir"] == "outputs/stage_j/paper_consistent"
    assert payload["historical_bridge_dir"] == "artifacts/stage_j_qwen_redesign_standard"
    assert payload["completion_statuses"] == ["export_visible_complete", "not_complete"]


def test_export_stage_j_paper_consistent_candidate_writes_canonical_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_j_qwen_redesign"
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
    (source_dir / "server" / "obfuscation_config.json").write_text(
        json.dumps(
            {
                "attention_profile": "rqk_hqk_block_taukv_taugroup",
                "lambda": 0.3,
                "h": 128,
                "alpha_e": 0.1,
                "alpha_h": 0.05,
                "adapted_layers": [0, 1],
                "beta": 0.25,
                "gamma": 0.75,
                "kappa_overrides": {"layers": {"0": {"input": 2.0, "post_attn": 3.0}}, "final": 4.0},
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
            "buffer::stage_a_model.model.layers.0.mlp.gate_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.up_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.down_weight": torch.zeros(12, 16),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_paper_consistent"
    result = export_stage_j_paper_consistent_candidate(export_dir, source_dir=source_dir)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["export_dir"] == export_dir
    assert manifest["track"] == "paper_consistent_candidate"
    assert manifest["standard_weight_proof"]["is_standard_weight_export"] is True
    assert manifest["bridge_is_acceptance_target"] is False
    assert manifest["candidate_role"] == "canonical_stage_j_acceptance_target"


def test_export_stage_j_paper_consistent_candidate_carries_component_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_j_qwen_redesign"
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
    (source_dir / "server" / "obfuscation_config.json").write_text(
        json.dumps(
            {
                "attention_profile": "rqk_hqk_block_taukv_taugroup",
                "lambda": 0.3,
                "h": 128,
                "adapted_layers": [0, 1, 2],
                "beta": 0.25,
                "gamma": 0.75,
                "kappa_overrides": {"layers": {"0": {"input": 2.0, "post_attn": 3.0}}, "final": 4.0},
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
            "buffer::stage_a_model.model.layers.0.mlp.gate_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.up_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.down_weight": torch.zeros(12, 16),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_paper_consistent"
    export_stage_j_paper_consistent_candidate(export_dir, source_dir=source_dir)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_visible_components"]["attention"]["profile"] == "rqk_hqk_block_taukv_taugroup"
    assert manifest["export_visible_components"]["attention"]["has_profile"] is True
    assert manifest["export_visible_components"]["attention"]["has_head_group_semantics"] is True
    assert manifest["export_visible_components"]["attention"]["has_block_semantics"] is True
    assert manifest["export_visible_components"]["ffn"]["adapted_layers_count"] == 3
    assert manifest["export_visible_components"]["norm"]["strategy"] == "kappa_fused"
    assert manifest["export_visible_components"]["norm"]["has_kappa_overrides"] is True


def test_export_visible_metadata_marks_kappa_fused_calibratable_without_overrides(tmp_path: Path) -> None:
    source_server = tmp_path / "server"
    source_server.mkdir(parents=True)
    (source_server / "obfuscation_config.json").write_text(
        json.dumps(
            {
                "attention_profile": "rqk_hqk_block_taukv_taugroup",
                "model_dir": "Qwen/Qwen2.5-0.5B",
                "seed": 7,
                "lambda": 0.3,
                "h": 128,
                "prompts_for_kappa": ["hello"],
                "adapted_layers": [0, 1],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = _build_export_visible_component_metadata(source_server, "kappa_fused")
    assert metadata["norm"]["has_kappa_overrides"] is True


def test_build_stage_j_paper_consistent_evidence_bundle_writes_all_required_reports(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "artifacts" / "stage_j_qwen_paper_consistent"
    source_dir = tmp_path / "artifacts" / "stage_j_qwen_redesign"
    (candidate_dir / "server").mkdir(parents=True)
    (candidate_dir / "client").mkdir(parents=True)
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "track": "paper_consistent_candidate",
                "norm_strategy": "kappa_fused",
                "standard_weight_proof": {"is_standard_weight_export": True, "layout": "standard_weight_visible"},
                "export_visible_components": {
                    "attention": {"profile": "rqk_hqk_block_taukv_taugroup", "has_profile": True, "has_head_group_semantics": True, "has_block_semantics": True},
                    "ffn": {"adapted_layers_count": 2, "beta": 0.25, "gamma": 0.75},
                    "norm": {"strategy": "kappa_fused", "has_kappa_overrides": True},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 12),
            "lm_head.weight": torch.zeros(8, 12),
        },
        str(candidate_dir / "server" / "model.safetensors"),
    )
    bundle = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=candidate_dir,
        source_dir=source_dir,
        output_dir=tmp_path / "outputs" / "stage_j" / "paper_consistent",
        correctness_payload={
            "summary": {
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
                "avg_restored_full_logits_max_abs_error": 0.0,
            }
        },
    )

    assert bundle["standard_weight_proof"]["status"] == "pass"
    assert bundle["attention_export_visible_proof"]["status"] == "pass"
    assert bundle["ffn_export_visible_proof"]["status"] == "pass"
    assert bundle["norm_export_visible_proof"]["status"] == "pass"
    assert bundle["correctness_regression"]["status"] == "pass"


def test_build_stage_j_paper_consistent_evidence_bundle_requires_current_candidate_weights(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "artifacts" / "stage_j_qwen_paper_consistent"
    source_dir = tmp_path / "artifacts" / "stage_j_qwen_redesign"
    (candidate_dir / "server").mkdir(parents=True)
    (candidate_dir / "client").mkdir(parents=True)
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "track": "paper_consistent_candidate",
                "norm_strategy": "kappa_fused",
                "standard_weight_proof": {"is_standard_weight_export": True, "layout": "standard_weight_visible"},
                "export_visible_components": {
                    "attention": {"profile": "rqk_hqk_block_taukv_taugroup", "has_profile": True, "has_head_group_semantics": True, "has_block_semantics": True},
                    "ffn": {"adapted_layers_count": 2, "beta": 0.25, "gamma": 0.75},
                    "norm": {"strategy": "kappa_fused", "has_kappa_overrides": True},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=candidate_dir,
        source_dir=source_dir,
        output_dir=tmp_path / "outputs" / "stage_j" / "paper_consistent",
        correctness_payload={
            "summary": {
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
                "avg_restored_full_logits_max_abs_error": 0.0,
            }
        },
    )

    assert bundle["standard_weight_proof"]["status"] == "fail"
    assert bundle["standard_weight_proof"]["layout"] == "missing_model_safetensors"
    assert bundle["completion_summary"]["completion_status"] == "not_complete"
    assert "standard_weight_proof" in bundle["completion_summary"]["blocking_components"]


def test_build_stage_j_paper_consistent_completion_summary_returns_export_visible_complete() -> None:
    summary = build_stage_j_paper_consistent_completion_summary(
        {
            "standard_weight_proof": {"status": "pass"},
            "attention_export_visible_proof": {"status": "pass"},
            "ffn_export_visible_proof": {"status": "pass"},
            "norm_export_visible_proof": {"status": "pass"},
            "correctness_regression": {"status": "pass"},
        }
    )
    assert summary["completion_status"] == "export_visible_complete"
    assert summary["blocking_components"] == []


def test_stage_j_paper_consistent_bundle_reports_explicit_completion_state(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "artifacts" / "stage_j_qwen_paper_consistent"
    source_dir = tmp_path / "artifacts" / "stage_j_qwen_redesign"
    output_dir = tmp_path / "outputs" / "stage_j" / "paper_consistent"
    (candidate_dir / "server").mkdir(parents=True)
    (candidate_dir / "client").mkdir(parents=True)
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "track": "paper_consistent_candidate",
                "standard_weight_proof": {"is_standard_weight_export": True, "layout": "standard_weight_visible"},
                "export_visible_components": {
                    "attention": {"profile": "rqk_hqk_block_taukv_taugroup", "has_profile": True, "has_head_group_semantics": True, "has_block_semantics": True},
                    "ffn": {"adapted_layers_count": 2, "beta": 0.25, "gamma": 0.75},
                    "norm": {"strategy": "kappa_fused", "has_kappa_overrides": True},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 12),
            "lm_head.weight": torch.zeros(8, 12),
        },
        str(candidate_dir / "server" / "model.safetensors"),
    )

    bundle = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=candidate_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        correctness_payload={
            "summary": {
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
                "avg_restored_full_logits_max_abs_error": 0.0,
            }
        },
    )

    assert bundle["completion_summary"]["completion_status"] == "export_visible_complete"
    assert (output_dir / "completion_summary.json").exists()
