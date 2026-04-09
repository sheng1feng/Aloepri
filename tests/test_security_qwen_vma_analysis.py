from src.security_qwen import (
    build_vma_layer_ablation_payload,
    build_vma_source_attribution_payload,
    infer_vma_default_projection_layers,
)


def test_vma_source_attribution_payload_preserves_labels() -> None:
    payload = build_vma_source_attribution_payload(
        target_name="stage_j_stable_reference",
        result_payloads=[
            {
                "config": {
                    "attribution_label": "direct_only",
                    "include_direct_sources": True,
                    "projection_kinds": [],
                    "layer_indices": [],
                },
                "metrics": {
                    "token_top1_recovery_rate": 0.1,
                    "token_top10_recovery_rate": 0.2,
                    "projection_source_count": 0,
                    "total_source_count": 2,
                },
                "summary": {"risk_level": "medium"},
            },
            {
                "config": {
                    "attribution_label": "projection_only_q",
                    "include_direct_sources": False,
                    "projection_kinds": ["q"],
                    "layer_indices": [0, 11, 23],
                },
                "metrics": {
                    "token_top1_recovery_rate": 0.3,
                    "token_top10_recovery_rate": 0.4,
                    "projection_source_count": 3,
                    "total_source_count": 3,
                },
                "summary": {"risk_level": "high"},
            },
        ],
    )
    assert payload["format"] == "qwen_security_vma_source_attribution_v1"
    assert payload["rows"][0]["label"] == "projection_only_q"


def test_vma_layer_ablation_payload_preserves_layer_labels() -> None:
    payload = build_vma_layer_ablation_payload(
        target_name="stage_h_full_obfuscated",
        result_payloads=[
            {
                "config": {
                    "ablation_label": "layer_0_only",
                    "layer_indices": [0],
                    "include_direct_sources": True,
                    "projection_kinds": ["q", "k"],
                },
                "metrics": {
                    "token_top1_recovery_rate": 0.05,
                    "token_top10_recovery_rate": 0.2,
                    "projection_source_count": 2,
                },
                "summary": {"risk_level": "low"},
            }
        ],
    )
    assert payload["format"] == "qwen_security_vma_layer_ablation_v1"
    assert payload["rows"][0]["label"] == "layer_0_only"


def test_infer_vma_default_projection_layers_returns_three_layers_for_stage_j() -> None:
    layers = infer_vma_default_projection_layers(target_name="stage_j_stable_reference")
    assert len(layers) == 3
    assert layers[0] == 0
