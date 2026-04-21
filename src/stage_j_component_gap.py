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
            "status": "unresolved",
            "notes": "Current bridge still uses placeholder or heuristic norm materialization and remains strongly non-equivalent.",
        },
    }
