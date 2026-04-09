from __future__ import annotations


def classify_risk_level(primary_metric_value: float | None) -> str:
    if primary_metric_value is None:
        return "unknown"
    if primary_metric_value >= 0.3:
        return "high"
    if primary_metric_value >= 0.1:
        return "medium"
    return "low"
