from src.stage_j_bridge_scan import rank_bridge_strategies


def test_rank_bridge_strategies_prefers_lower_error_then_match_rate() -> None:
    ranked = rank_bridge_strategies(
        [
            {
                "norm_strategy": "a",
                "summary": {
                    "avg_restored_full_logits_max_abs_error": 10.0,
                    "generated_ids_exact_match_rate": 0.0,
                },
            },
            {
                "norm_strategy": "b",
                "summary": {
                    "avg_restored_full_logits_max_abs_error": 5.0,
                    "generated_ids_exact_match_rate": 0.0,
                },
            },
        ]
    )
    assert ranked[0]["norm_strategy"] == "b"
