from __future__ import annotations

from typing import Any


def build_stage_j_component_gap_report() -> dict[str, Any]:
    return {
        "embed_head": {
            "status": "partially_resolved",
            "notes": "Standard-visible bridge already materializes embed/lm_head into model.* keys.",
        },
        "attention": {
            "status": "partially_resolved",
            "notes": "Bridge maps qkv/o to standard keys, but full paper-consistent expression retention is not yet proven.",
        },
        "ffn": {
            "status": "partially_resolved",
            "notes": "Bridge maps gate/up/down to standard keys, but expression equivalence is not yet proven.",
        },
        "norm": {
            "status": "partially_resolved",
            "notes": "kappa_fused currently outperforms ones and metric_diag_sqrt, but bridge regression still shows a large remaining gap.",
        },
    }
