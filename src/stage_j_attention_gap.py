from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def summarize_attention_metadata(
    *,
    q_feature_inv_order: torch.Tensor,
    kv_feature_inv_order: torch.Tensor,
    q_dense_inverse: torch.Tensor,
    k_dense_inverse: torch.Tensor,
) -> dict[str, Any]:
    q_feature_inv_order = q_feature_inv_order.to(torch.long)
    kv_feature_inv_order = kv_feature_inv_order.to(torch.long)
    q_dense_inverse = q_dense_inverse.to(torch.float32)
    k_dense_inverse = k_dense_inverse.to(torch.float32)

    q_identity = torch.eye(q_dense_inverse.shape[0], dtype=torch.float32)
    k_identity = torch.eye(k_dense_inverse.shape[0], dtype=torch.float32)
    return {
        "q_order_is_identity": bool(torch.equal(q_feature_inv_order, torch.arange(q_feature_inv_order.numel()))),
        "kv_order_is_identity": bool(torch.equal(kv_feature_inv_order, torch.arange(kv_feature_inv_order.numel()))),
        "q_dense_identity_max_abs_error": float((q_dense_inverse - q_identity).abs().max().item()),
        "k_dense_identity_max_abs_error": float((k_dense_inverse - k_identity).abs().max().item()),
    }


def build_stage_j_attention_gap_report(
    server_dir: str | Path = "artifacts/stage_j_qwen_redesign/server",
) -> dict[str, Any]:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    payload = summarize_attention_metadata(
        q_feature_inv_order=state["buffer::stage_a_model.model.layers.0.self_attn.q_feature_inv_order"],
        kv_feature_inv_order=state["buffer::stage_a_model.model.layers.0.self_attn.kv_feature_inv_order"],
        q_dense_inverse=state["buffer::stage_a_model.model.layers.0.self_attn.q_dense_inverse"],
        k_dense_inverse=state["buffer::stage_a_model.model.layers.0.self_attn.k_dense_inverse"],
    )
    unresolved = []
    if not payload["q_order_is_identity"]:
        unresolved.append("q_feature_order")
    if not payload["kv_order_is_identity"]:
        unresolved.append("kv_feature_order")
    if payload["q_dense_identity_max_abs_error"] > 1e-6:
        unresolved.append("q_dense")
    if payload["k_dense_identity_max_abs_error"] > 1e-6:
        unresolved.append("k_dense")
    return {
        "stage": "J",
        "server_dir": str(server_dir),
        "metrics": payload,
        "summary": {
            "unresolved_attention_sites": unresolved,
            "all_attention_metadata_identity": not bool(unresolved),
        },
    }
