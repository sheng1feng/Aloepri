from __future__ import annotations

from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


def build_ia_template(target: SecurityEvalTarget) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="ia",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_smoke_prompts",
            "seed": 20260323,
            "variants": ["gate_ia", "attn_ia"],
        },
        metrics={
            "token_top1_recovery_rate": primary_metric,
            "invariant_separability_score": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; IA implementation not started.",
        },
        artifacts={},
    )
