import torch

from src.security_qwen.ima import (
    _evaluate_inversion_predictions,
    _fit_ridge_regressor,
    _predict_ridge,
    build_ima_comparison_payload,
)


def test_ridge_regressor_recovers_identity_mapping() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y = x.clone()
    model = _fit_ridge_regressor(x, y, ridge_alpha=1e-4)
    pred = _predict_ridge(model, x)
    assert torch.allclose(pred, y, atol=1e-3)


def test_evaluate_inversion_predictions_reports_hits() -> None:
    baseline_embed = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    predicted = baseline_embed[:2].clone()
    metrics = _evaluate_inversion_predictions(
        predicted_embeddings=predicted,
        true_plain_ids=torch.tensor([0, 1], dtype=torch.long),
        candidate_plain_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        baseline_embed=baseline_embed,
        topk=2,
    )
    assert metrics["token_top1_recovery_rate"] == 1.0
    assert metrics["token_top10_recovery_rate"] == 1.0


def test_ima_comparison_payload_computes_stage_a_deltas() -> None:
    payload = build_ima_comparison_payload(
        result_payloads=[
            {
                "target": {"stage": "A", "profile": None, "artifact_dir": "artifacts/stage_i_vllm", "variant": "stage_a_vocab_permutation"},
                "metrics": {"token_top1_recovery_rate": 0.9, "token_top10_recovery_rate": 0.95, "embedding_cosine_similarity": 0.99, "sensitive_token_recovery_rate": 0.8},
                "summary": {"risk_level": "high", "status": "completed"},
                "config": {"selected_ridge_alpha": 0.01},
                "artifacts": {"resolved_target": {"name": "stage_a_standard"}},
            },
            {
                "target": {"stage": "H", "profile": None, "artifact_dir": "artifacts/stage_h_full_obfuscated", "variant": "keymat_full_obfuscated"},
                "metrics": {"token_top1_recovery_rate": 0.2, "token_top10_recovery_rate": 0.4, "embedding_cosine_similarity": 0.5, "sensitive_token_recovery_rate": 0.1},
                "summary": {"risk_level": "medium", "status": "completed"},
                "config": {"selected_ridge_alpha": 1.0},
                "artifacts": {"resolved_target": {"name": "stage_h_full_obfuscated"}},
            },
        ]
    )
    assert payload["format"] == "qwen_security_ima_comparison_v1"
    row_map = {row["target_name"]: row for row in payload["rows"]}
    assert abs(row_map["stage_h_full_obfuscated"]["vs_stage_a_top1_delta"] + 0.7) < 1e-8
