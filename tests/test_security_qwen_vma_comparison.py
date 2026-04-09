from src.security_qwen import build_vma_comparison_payload


def test_vma_comparison_payload_computes_stage_a_deltas() -> None:
    payload = build_vma_comparison_payload(
        result_payloads=[
            {
                "target": {"stage": "A", "profile": None, "artifact_dir": "artifacts/stage_i_vllm", "variant": "stage_a_vocab_permutation"},
                "metrics": {"token_top1_recovery_rate": 0.5, "token_top10_recovery_rate": 0.7, "token_top100_recovery_rate": 1.0, "sensitive_token_recovery_rate": 0.4},
                "summary": {"risk_level": "high", "status": "completed"},
                "artifacts": {"resolved_target": {"name": "stage_a_standard"}},
            },
            {
                "target": {"stage": "J", "profile": "stable_reference", "artifact_dir": "artifacts/stage_j_full_square", "variant": "standard_shape_full_layer"},
                "metrics": {"token_top1_recovery_rate": 0.2, "token_top10_recovery_rate": 0.3, "token_top100_recovery_rate": 0.8, "sensitive_token_recovery_rate": 0.1},
                "summary": {"risk_level": "medium", "status": "completed"},
                "artifacts": {"resolved_target": {"name": "stage_j_stable_reference"}},
            },
        ]
    )
    assert payload["format"] == "qwen_security_vma_comparison_v1"
    row_map = {row["target_name"]: row for row in payload["rows"]}
    assert row_map["stage_a_standard"]["vs_stage_a_top1_delta"] == 0.0
    assert abs(row_map["stage_j_stable_reference"]["vs_stage_a_top1_delta"] + 0.3) < 1e-8
