from pathlib import Path

import torch

from src.security_qwen.ima import (
    _evaluate_inversion_predictions,
    _fit_ridge_regressor,
    _load_public_inversion_texts,
    _predict_ridge,
    build_ima_comparison_payload,
    build_paper_like_inverter_config,
    default_ima_output_path,
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


def test_default_ima_output_path_uses_mode_specific_suffix() -> None:
    assert default_ima_output_path("stage_k_default") == Path("outputs/security_qwen/ima/stage_k_default.json")
    assert default_ima_output_path("stage_k_default", mode="paper_like") == Path("outputs/security_qwen/ima/stage_k_default.paper_like.json")


def test_load_public_inversion_texts_keeps_non_empty_repo_like_docs(tmp_path: Path) -> None:
    first = tmp_path / "paper.txt"
    second = tmp_path / "notes.md"
    blank = tmp_path / "blank.md"
    first.write_text("AloePri paper corpus line one.\n\nline two.", encoding="utf-8")
    second.write_text("# Heading\n\nPaper-like inversion notes.", encoding="utf-8")
    blank.write_text("   \n", encoding="utf-8")

    texts = _load_public_inversion_texts([str(first), str(second), str(blank), str(tmp_path / "missing.md")])
    assert texts == [
        "AloePri paper corpus line one.\n\nline two.",
        "# Heading\n\nPaper-like inversion notes.",
    ]


def test_build_paper_like_inverter_config_matches_paper_shape() -> None:
    config = build_paper_like_inverter_config(
        observed_hidden_size=1152,
        vocab_size=151936,
    )
    assert config.hidden_size == 1152
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 8
    assert config.vocab_size == 151936
    assert str(config.torch_dtype) == "float32"
