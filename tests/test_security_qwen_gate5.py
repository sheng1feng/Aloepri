from src.security_qwen.gate5_scan import build_gate5_scan_payload, default_gate5_cases


def test_default_gate5_cases_cover_representative_points() -> None:
    names = [item.name for item in default_gate5_cases()]
    assert names == ["stable_reference", "tiny_a", "tiny_b", "small_a", "paper_like"]


def test_gate5_scan_payload_selects_recommended_nonzero_case() -> None:
    payload = build_gate5_scan_payload(
        [
            {
                "name": "stable_reference",
                "alpha_e": 0.0,
                "alpha_h": 0.0,
                "generated_ids_exact_match_rate": 1.0,
                "avg_final_logits_restored_max_abs_error": 0.0,
                "vma_projection_top1": 0.9,
                "ima_top1": 0.9,
                "isa_hidden_top1": 0.1,
                "tfma_domain_top10": 0.7,
                "sda_distribution_bleu4": 0.1,
            },
            {
                "name": "tiny_a",
                "alpha_e": 0.02,
                "alpha_h": 0.01,
                "generated_ids_exact_match_rate": 1.0,
                "avg_final_logits_restored_max_abs_error": 0.6,
                "vma_projection_top1": 0.5,
                "ima_top1": 0.6,
                "isa_hidden_top1": 0.05,
                "tfma_domain_top10": 0.7,
                "sda_distribution_bleu4": 0.1,
            },
            {
                "name": "small_a",
                "alpha_e": 0.1,
                "alpha_h": 0.05,
                "generated_ids_exact_match_rate": 0.6,
                "avg_final_logits_restored_max_abs_error": 3.4,
                "vma_projection_top1": 0.2,
                "ima_top1": 0.6,
                "isa_hidden_top1": 0.05,
                "tfma_domain_top10": 0.7,
                "sda_distribution_bleu4": 0.1,
            },
        ]
    )
    assert payload["format"] == "qwen_security_gate5_scan_v1"
    assert payload["recommended_nonzero_case"]["name"] == "tiny_a"
