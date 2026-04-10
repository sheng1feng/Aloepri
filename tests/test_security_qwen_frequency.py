from src.security_qwen import (
    build_frequency_corpora,
    build_sda_comparison_payload,
    build_tfma_comparison_payload,
)


def test_frequency_corpora_provide_three_knowledge_settings() -> None:
    for setting in ["zero_knowledge", "domain_aware", "distribution_aware"]:
        corpora = build_frequency_corpora(setting)
        assert len(corpora["reference_texts"]) > 0
        assert len(corpora["private_texts"]) > 0


def test_tfma_comparison_payload_keeps_knowledge_settings() -> None:
    payload = build_tfma_comparison_payload(
        result_payloads=[
            {
                "target": {"stage": "A", "profile": None, "artifact_dir": "artifacts/stage_i_vllm", "variant": "stage_a_vocab_permutation"},
                "metrics": {"token_top1_recovery_rate": 0.2, "token_top10_recovery_rate": 0.5, "token_top100_recovery_rate": 0.9, "sensitive_token_recovery_rate": 0.1},
                "summary": {"risk_level": "medium", "status": "completed"},
                "config": {"knowledge_setting": "zero_knowledge"},
                "artifacts": {"resolved_target": {"name": "stage_a_standard"}},
            },
            {
                "target": {"stage": "J", "profile": "stable_reference", "artifact_dir": "artifacts/stage_j_full_square", "variant": "standard_shape_full_layer"},
                "metrics": {"token_top1_recovery_rate": 0.1, "token_top10_recovery_rate": 0.3, "token_top100_recovery_rate": 0.8, "sensitive_token_recovery_rate": 0.0},
                "summary": {"risk_level": "medium", "status": "completed"},
                "config": {"knowledge_setting": "zero_knowledge"},
                "artifacts": {"resolved_target": {"name": "stage_j_stable_reference"}},
            },
        ]
    )
    assert payload["format"] == "qwen_security_tfma_comparison_v1"
    assert payload["rows"][0]["knowledge_setting"] == "zero_knowledge"


def test_sda_comparison_payload_keeps_bleu_deltas() -> None:
    payload = build_sda_comparison_payload(
        result_payloads=[
            {
                "target": {"stage": "A", "profile": None, "artifact_dir": "artifacts/stage_i_vllm", "variant": "stage_a_vocab_permutation"},
                "metrics": {"token_top1_recovery_rate": 0.2, "token_top10_recovery_rate": 0.5, "token_top100_recovery_rate": 0.9, "sensitive_token_recovery_rate": 0.1, "bleu4": 0.6},
                "summary": {"risk_level": "medium", "status": "completed"},
                "config": {"knowledge_setting": "domain_aware"},
                "artifacts": {"resolved_target": {"name": "stage_a_standard"}},
            },
            {
                "target": {"stage": "J", "profile": "stable_reference", "artifact_dir": "artifacts/stage_j_full_square", "variant": "standard_shape_full_layer"},
                "metrics": {"token_top1_recovery_rate": 0.1, "token_top10_recovery_rate": 0.3, "token_top100_recovery_rate": 0.8, "sensitive_token_recovery_rate": 0.0, "bleu4": 0.2},
                "summary": {"risk_level": "low", "status": "completed"},
                "config": {"knowledge_setting": "domain_aware"},
                "artifacts": {"resolved_target": {"name": "stage_j_stable_reference"}},
            },
        ]
    )
    assert payload["format"] == "qwen_security_sda_comparison_v1"
    assert abs(payload["rows"][1]["vs_stage_a_bleu4_delta"] + 0.4) < 1e-8
