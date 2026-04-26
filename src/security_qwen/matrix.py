from __future__ import annotations

from typing import Any

from src.security_qwen.artifacts import default_security_targets
from src.security_qwen.datasets import phase0_dataset_payload


def security_matrix_payload() -> dict[str, Any]:
    targets = {item.name: item for item in default_security_targets()}
    return {
        "format": "qwen_security_matrix_v1",
        "phase": "phase0",
        "datasets": phase0_dataset_payload()["datasets"],
        "matrices": {
            "static_weight_attacks": {
                "attacks": ["vma", "ia"],
                "targets": [
                    targets["stage_h_full_obfuscated"].name,
                    targets["stage_j_stable_reference"].name,
                    targets["stage_j_tiny_a"].name,
                    targets["stage_k_reference"].name,
                    targets["stage_k_default"].name,
                ],
            },
            "training_inversion_attacks": {
                "attacks": ["ima"],
                "targets": [
                    targets["stage_a_standard"].name,
                    targets["stage_h_full_obfuscated"].name,
                    targets["stage_j_stable_reference"].name,
                    targets["stage_j_tiny_a"].name,
                ],
            },
            "deployment_intermediate_attacks": {
                "attacks": ["isa"],
                "observable_types": ["attention_score", "hidden_state", "layer_output", "kv_cache"],
                "targets": [
                    targets["stage_h_full_obfuscated"].name,
                    targets["stage_j_stable_reference"].name,
                    targets["stage_j_tiny_a"].name,
                    targets["stage_k_reference"].name,
                    targets["stage_k_default"].name,
                ],
            },
            "online_frequency_attacks": {
                "attacks": ["tfma", "sda"],
                "knowledge_settings": ["zero_knowledge", "domain_aware", "distribution_aware"],
                "targets": [
                    targets["stage_j_stable_reference"].name,
                    targets["stage_j_tiny_a"].name,
                    targets["stage_k_reference"].name,
                    targets["stage_k_default"].name,
                ],
            },
        },
    }
