from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.key_manager import validate_permutation
from src.model_loader import resolve_torch_dtype
from src.stage_h import build_stage_h_deployable_inventory


def _normalize_saved_tokenizer_config(server_dir: Path) -> None:
    tokenizer_config_path = server_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return
    payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    extra_special_tokens = payload.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        payload.pop("extra_special_tokens", None)
        tokenizer_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_stage_i_manifest() -> dict[str, Any]:
    return {
        "format": "stage_i_vllm_v1",
        "server_dir": "server",
        "client_dir": "client",
        "server_files": [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
        "client_files": [
            "client_secret.pt",
        ],
    }


def summarize_token_partitions(
    *,
    tokenizer: Any,
    model_vocab_size: int,
    perm_vocab: torch.Tensor,
) -> dict[str, Any]:
    perm_vocab = torch.as_tensor(perm_vocab, dtype=torch.long).cpu()
    special_ids = sorted(token_id for token_id in tokenizer.all_special_ids if token_id < tokenizer.vocab_size)
    tail_start = int(tokenizer.vocab_size)
    tail_ids = list(range(tail_start, model_vocab_size))
    tail_tensor = torch.arange(tail_start, model_vocab_size, dtype=torch.long)

    special_fixed = all(int(perm_vocab[token_id].item()) == token_id for token_id in special_ids)
    tail_fixed = True
    if tail_ids:
        tail_fixed = bool(torch.equal(perm_vocab[tail_start:model_vocab_size], tail_tensor))

    return {
        "perm_is_valid": validate_permutation(perm_vocab),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "model_vocab_size": int(model_vocab_size),
        "special_token_count": len(special_ids),
        "special_ids_fixed": special_fixed,
        "tail_row_count": max(model_vocab_size - int(tokenizer.vocab_size), 0),
        "tail_rows_fixed": tail_fixed,
    }


def export_stage_i_vllm_checkpoint(
    export_dir: str | Path,
    *,
    tokenizer: Any,
    stage_a_model: Any,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    server_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(server_dir)
    _normalize_saved_tokenizer_config(server_dir)
    stage_a_model.config.save_pretrained(server_dir)
    try:
        stage_a_model.generation_config.save_pretrained(server_dir)
    except Exception:
        GenerationConfig.from_model_config(stage_a_model.config).save_pretrained(server_dir)
    stage_a_model.save_pretrained(server_dir, safe_serialization=True)

    torch.save(
        {
            "perm_vocab": torch.as_tensor(perm_vocab, dtype=torch.long).cpu(),
            "inv_perm_vocab": torch.as_tensor(inv_perm_vocab, dtype=torch.long).cpu(),
            "metadata": metadata,
        },
        client_dir / "client_secret.pt",
    )
    (export_dir / "manifest.json").write_text(
        json.dumps(build_stage_i_manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "stage_i_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "export_dir": export_dir,
        "server_dir": server_dir,
        "client_secret_path": client_dir / "client_secret.pt",
    }


def load_stage_i_hf_bundle(
    server_dir: str | Path,
    *,
    client_secret_path: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "auto",
) -> dict[str, Any]:
    server_dir = Path(server_dir)
    tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        server_dir,
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model.eval()
    if device != "cpu":
        model.to(device)

    if client_secret_path is None:
        default_secret = server_dir.parent / "client" / "client_secret.pt"
        if default_secret.exists():
            client_secret_path = default_secret

    secret: dict[str, Any] | None = None
    perm_vocab = None
    inv_perm_vocab = None
    if client_secret_path is not None and Path(client_secret_path).exists():
        secret = torch.load(client_secret_path, map_location="cpu")
        perm_vocab = torch.as_tensor(secret["perm_vocab"], dtype=torch.long)
        inv_perm_vocab = torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long)

    metadata_path = server_dir.parent / "stage_i_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "tokenizer": tokenizer,
        "model": model,
        "perm_vocab": perm_vocab,
        "inv_perm_vocab": inv_perm_vocab,
        "secret": secret,
        "metadata": metadata,
    }


def build_phase2_feasibility_summary() -> dict[str, Any]:
    return {
        "phase": "stage_i_phase2_feasibility",
        "scope": "Assess what can be materialized into a standard HF/vLLM-compatible checkpoint without custom kernels.",
        "components": [
            {
                "component": "Stage A vocab permutation",
                "status": "feasible_now",
                "requires_runtime_custom_logic": False,
                "affects_standard_kernels": False,
                "rationale": "Pure embedding/lm_head row permutation with client-side token/logit remapping.",
            },
            {
                "component": "Embedding/head noise + KeyMat",
                "status": "blocked_by_expanded_hidden_dim",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Current Algorithm-1 KeyMat expands hidden size, so obfuscated embed/head weights no longer match the original HF architecture shapes.",
            },
            {
                "component": "FFN fused path",
                "status": "blocked_by_expanded_hidden_dim",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Fused FFN preserves MLP semantics but its input/output dimensions follow expanded KeyMat hidden states, so standard Qwen MLP shapes no longer match.",
            },
            {
                "component": "Attention staticized path",
                "status": "blocked_by_expanded_hidden_dim",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Stage H reduced runtime structure, but q/k/v/o weights are still expressed on expanded hidden states and cannot be copied into the original Qwen attention parameter shapes.",
            },
            {
                "component": "RMSNorm fused path",
                "status": "not_vllm_ready",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Metric-matrix style fused norm changes standard RMSNorm kernel semantics and would need rollback or custom kernel support.",
            },
        ],
        "redesigned_stage_i": build_stage_i_deployability_matrix(),
    }


def build_stage_i_deployability_matrix() -> dict[str, Any]:
    inventory = build_stage_h_deployable_inventory()
    return {
        "stage": "I",
        "source_stage": "H",
        "inventory_goal": inventory["goal"],
        "runtime_boundary": {
            "standard_transformer_graph": True,
            "custom_online_operator_required": False,
            "compatible_target_surfaces": ["transformers", "vllm", "sglang"],
        },
        "validated_components": {
            "embedding_head": "standard_weight_rewrite",
            "attention_diversity": "needs_materialization_check",
            "ffn_component_transform": "needs_materialization_check",
            "norm_kappa_correction": "supported_if_fused_offline",
        },
        "legacy_reference": {
            "legacy_stage_i_scope": "stage_a_standard_entry_and_phase2_probe",
            "legacy_report": "docs/阶段I_vLLM复现报告.md",
        },
    }
