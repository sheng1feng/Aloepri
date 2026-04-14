from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors.torch import load_file


REQUIRED_STANDARD_KEYS = [
    "model.embed_tokens.weight",
    "lm_head.weight",
]


def build_stage_j_standard_weight_proof(server_dir: str | Path) -> dict[str, Any]:
    server_dir = Path(server_dir)
    model_path = server_dir / "model.safetensors"
    if not model_path.exists():
        return {
            "is_standard_weight_export": False,
            "layout": "missing_model_safetensors",
            "standard_key_count": 0,
            "buffered_key_count": 0,
            "missing_standard_keys": list(REQUIRED_STANDARD_KEYS),
            "sample_standard_keys": [],
            "sample_buffered_keys": [],
        }

    state = load_file(str(model_path))
    keys = list(state.keys())
    standard_keys = [key for key in keys if key.startswith("model.") or key.startswith("lm_head.")]
    buffered_keys = [key for key in keys if key.startswith("buffer::stage_a_model.")]
    missing_standard_keys = [key for key in REQUIRED_STANDARD_KEYS if key not in state]

    is_standard_weight_export = len(missing_standard_keys) == 0 and len(buffered_keys) == 0
    layout = "standard_weight_visible" if is_standard_weight_export else "buffered_stage_style"

    return {
        "is_standard_weight_export": is_standard_weight_export,
        "layout": layout,
        "standard_key_count": len(standard_keys),
        "buffered_key_count": len(buffered_keys),
        "missing_standard_keys": missing_standard_keys,
        "sample_standard_keys": standard_keys[:20],
        "sample_buffered_keys": buffered_keys[:20],
    }
