from src.stage_j_bridge_regression import summarize_bridge_items


def test_summarize_bridge_items_reports_mean_metrics() -> None:
    payload = summarize_bridge_items(
        [
            {
                "restored_full_logits_max_abs_error": 1.0,
                "restored_full_logits_mean_abs_error": 0.5,
                "restored_last_token_max_abs_error": 0.25,
                "restored_last_token_mean_abs_error": 0.1,
                "generated_ids_exact_match": True,
                "generated_text_exact_match": False,
            },
            {
                "restored_full_logits_max_abs_error": 3.0,
                "restored_full_logits_mean_abs_error": 1.5,
                "restored_last_token_max_abs_error": 0.75,
                "restored_last_token_mean_abs_error": 0.3,
                "generated_ids_exact_match": False,
                "generated_text_exact_match": False,
            },
        ]
    )
    assert payload["prompt_count"] == 2
    assert payload["avg_restored_full_logits_max_abs_error"] == 2.0
    assert payload["generated_ids_exact_match_rate"] == 0.5
