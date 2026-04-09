from __future__ import annotations

from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


def build_isa_template(
    target: SecurityEvalTarget,
    *,
    observable_type: str = "hidden_state",
    observable_layer: str = "planned",
) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="isa",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_inversion_public_corpus",
            "seed": 20260323,
            "observable_type": observable_type,
            "observable_layer": observable_layer,
            "deployment_only": True,
        },
        metrics={
            "observable_type": observable_type,
            "observable_layer": observable_layer,
            "intermediate_top1_recovery_rate": primary_metric,
            "token_top10_recovery_rate": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "intermediate_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; ISA deployment-observable implementation not started.",
        },
        artifacts={},
    )
