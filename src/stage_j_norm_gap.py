from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def summarize_metric_matrix(matrix: torch.Tensor) -> dict[str, Any]:
    matrix = matrix.to(torch.float32)
    diag = torch.diagonal(matrix)
    offdiag = matrix - torch.diag_embed(diag)
    offdiag_norm = float(offdiag.norm().item())
    total_norm = float(matrix.norm().item())
    offdiag_ratio = 0.0 if total_norm == 0.0 else offdiag_norm / total_norm
    return {
        "shape": list(matrix.shape),
        "diag_mean": float(diag.mean().item()),
        "diag_max": float(diag.max().item()),
        "diag_min": float(diag.min().item()),
        "offdiag_fro_norm": offdiag_norm,
        "matrix_fro_norm": total_norm,
        "offdiag_ratio": offdiag_ratio,
        "standard_rmsnorm_equivalent": offdiag_ratio < 1e-6,
    }


def build_stage_j_norm_gap_report(
    server_dir: str | Path = "artifacts/stage_j_qwen_redesign/server",
) -> dict[str, Any]:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    keys = {
        "layer_0_input_norm": "buffer::stage_a_model.model.layers.0.input_layernorm.metric_matrix",
        "layer_0_post_attn_norm": "buffer::stage_a_model.model.layers.0.post_attention_layernorm.metric_matrix",
        "final_norm": "buffer::stage_a_model.model.norm.metric_matrix",
    }
    report = {name: summarize_metric_matrix(state[key]) for name, key in keys.items()}
    unresolved = [name for name, payload in report.items() if not payload["standard_rmsnorm_equivalent"]]
    return {
        "stage": "J",
        "server_dir": str(server_dir),
        "metrics": report,
        "summary": {
            "unresolved_norm_sites": unresolved,
            "all_standard_rmsnorm_equivalent": not bool(unresolved),
        },
    }
