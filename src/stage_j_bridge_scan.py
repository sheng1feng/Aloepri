from __future__ import annotations

from typing import Any


def rank_bridge_strategies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item["summary"].get("avg_restored_full_logits_max_abs_error", float("inf"))),
            -float(item["summary"].get("generated_ids_exact_match_rate", 0.0)),
        ),
    )
