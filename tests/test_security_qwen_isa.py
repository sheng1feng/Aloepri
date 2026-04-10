import torch

from src.security_qwen.isa import (
    _flatten_attention_score,
    _flatten_hidden_state,
    build_isa_comparison_payload,
)


def test_flatten_hidden_state_preserves_row_count() -> None:
    hidden = torch.randn(2, 4, 8)
    flat = _flatten_hidden_state(hidden)
    assert flat.shape == (8, 8)


def test_flatten_attention_score_preserves_row_count() -> None:
    attn = torch.randn(2, 3, 4, 4)
    flat = _flatten_attention_score(attn)
    assert flat.shape == (8, 12)


def test_isa_comparison_payload_computes_stage_a_deltas() -> None:
    payload = build_isa_comparison_payload(
        observable_type="hidden_state",
        result_payloads=[
            {
                "target": {"stage": "A", "profile": None, "artifact_dir": "artifacts/stage_i_vllm", "variant": "stage_a_vocab_permutation"},
                "metrics": {"observable_type": "hidden_state", "observable_layer": 23, "intermediate_top1_recovery_rate": 0.9, "token_top10_recovery_rate": 0.95, "embedding_cosine_similarity": 0.8, "sensitive_token_recovery_rate": 1.0},
                "summary": {"risk_level": "high", "status": "completed"},
                "config": {"selected_ridge_alpha": 0.01},
                "artifacts": {"resolved_target": {"name": "stage_a_standard"}},
            },
            {
                "target": {"stage": "H", "profile": None, "artifact_dir": "artifacts/stage_h_full_obfuscated", "variant": "keymat_full_obfuscated"},
                "metrics": {"observable_type": "hidden_state", "observable_layer": 23, "intermediate_top1_recovery_rate": 0.2, "token_top10_recovery_rate": 0.4, "embedding_cosine_similarity": 0.3, "sensitive_token_recovery_rate": 0.5},
                "summary": {"risk_level": "medium", "status": "completed"},
                "config": {"selected_ridge_alpha": 1.0},
                "artifacts": {"resolved_target": {"name": "stage_h_full_obfuscated"}},
            },
        ],
    )
    assert payload["format"] == "qwen_security_isa_comparison_v1"
    row_map = {row["target_name"]: row for row in payload["rows"]}
    assert abs(row_map["stage_h_full_obfuscated"]["vs_stage_a_top1_delta"] + 0.7) < 1e-8
