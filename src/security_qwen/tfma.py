from __future__ import annotations

from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


def build_tfma_template(
    target: SecurityEvalTarget,
    *,
    knowledge_setting: str = "zero_knowledge",
) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="tfma",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_frequency_attack_corpus",
            "seed": 20260323,
            "knowledge_setting": knowledge_setting,
        },
        metrics={
            "token_top10_recovery_rate": primary_metric,
            "token_top100_recovery_rate": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top10_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; TFMA implementation not started.",
        },
        artifacts={},
    )
