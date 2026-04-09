from __future__ import annotations

from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


def build_ima_template(target: SecurityEvalTarget) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="ima",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_inversion_public_corpus",
            "seed": 20260323,
            "prediction_targets": ["token_id", "embedding"],
        },
        metrics={
            "token_top1_recovery_rate": primary_metric,
            "token_top10_recovery_rate": None,
            "embedding_cosine_similarity": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; IMA implementation not started.",
        },
        artifacts={},
    )
