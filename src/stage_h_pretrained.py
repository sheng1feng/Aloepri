from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.model_loader import set_global_seed
from src.stage_b import TraceRecorder
from src.stage_f import build_default_stage_f_keymat, build_layer_stage_f_configs, calibrate_keymat_kappas
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model


def _normalize_saved_tokenizer_config(server_dir: Path) -> None:
    tokenizer_config_path = server_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return
    payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    extra_special_tokens = payload.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        payload.pop("extra_special_tokens", None)
        tokenizer_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_tensor_state(model) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for name, tensor in model.named_parameters():
        payload[f"param::{name}"] = tensor.detach().cpu().clone().contiguous()
    for name, tensor in model.named_buffers():
        payload[f"buffer::{name}"] = tensor.detach().cpu().clone().contiguous()
    return payload


def _load_tensor_state(model, tensor_state: dict[str, torch.Tensor]) -> None:
    parameter_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    for key, tensor in tensor_state.items():
        if key.startswith("param::"):
            name = key[len("param::") :]
            if name in parameter_map:
                parameter_map[name].data.copy_(tensor.to(device=parameter_map[name].device, dtype=parameter_map[name].dtype))
        elif key.startswith("buffer::"):
            name = key[len("buffer::") :]
            if name in buffer_map:
                buffer_map[name].data.copy_(tensor.to(device=buffer_map[name].device, dtype=buffer_map[name].dtype))


def export_stage_h_pretrained(
    export_dir: str | Path,
    *,
    tokenizer,
    baseline_model,
    stage_model,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    metadata: dict,
) -> None:
    export_dir = Path(export_dir)
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    server_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(server_dir)
    _normalize_saved_tokenizer_config(server_dir)
    baseline_model.config.save_pretrained(server_dir)
    try:
        baseline_model.generation_config.save_pretrained(server_dir)
    except Exception:
        GenerationConfig.from_model_config(baseline_model.config).save_pretrained(server_dir)

    save_file(_collect_tensor_state(stage_model), str(server_dir / "model.safetensors"))
    (server_dir / "obfuscation_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"perm_vocab": perm_vocab.cpu(), "inv_perm_vocab": inv_perm_vocab.cpu()}, client_dir / "client_secret.pt")
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "stage_h_pretrained_v1",
                "server_dir": "server",
                "client_dir": "client",
                "server_files": [
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "obfuscation_config.json",
                ],
                "client_files": ["client_secret.pt"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_stage_h_pretrained(
    server_dir: str | Path,
    *,
    client_secret_path: str | Path | None = None,
    recorder: TraceRecorder | None = None,
):
    server_dir = Path(server_dir)
    metadata = json.loads((server_dir / "obfuscation_config.json").read_text(encoding="utf-8"))
    set_global_seed(int(metadata["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(server_dir, trust_remote_code=True)
    baseline_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    baseline_model.eval()
    try:
        baseline_model.generation_config = GenerationConfig.from_pretrained(server_dir)
    except Exception:
        baseline_model.generation_config = GenerationConfig.from_model_config(config)

    adapted_layers = list(metadata["adapted_layers"])
    keymat_transform = build_default_stage_f_keymat(
        baseline_model,
        lam=float(metadata["lambda"]),
        h=int(metadata["h"]),
        seed=int(metadata["seed"]),
    )
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=metadata["prompts_for_kappa"],
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
        kappa_by_layer=kappas,
        attention_profile=metadata["attention_profile"],
        seed=int(metadata["seed"]),
        alpha_e=float(metadata.get("alpha_e", 0.0)),
        alpha_h=float(metadata.get("alpha_h", 0.0)),
    )
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    stage_model, perm_vocab, inv_perm_vocab = build_stage_h_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        keymat_transform=keymat_transform,
        seed=int(metadata["seed"]),
        recorder=recorder,
        layer_configs=layer_configs_h,
        adapted_layers=adapted_layers,
        alpha_e=float(metadata.get("alpha_e", 0.0)),
        alpha_h=float(metadata.get("alpha_h", 0.0)),
        use_keymat_head=True,
        beta=int(metadata.get("beta", 8)),
        gamma=float(metadata.get("gamma", 1e3)),
    )
    _load_tensor_state(stage_model, load_file(str(server_dir / "model.safetensors"), device="cpu"))

    if client_secret_path is None:
        default_client = server_dir.parent / "client" / "client_secret.pt"
        if default_client.exists():
            client_secret_path = default_client
    if client_secret_path is not None and Path(client_secret_path).exists():
        secret = torch.load(client_secret_path, map_location="cpu")
        perm_vocab = secret["perm_vocab"]
        inv_perm_vocab = secret["inv_perm_vocab"]

    return {
        "metadata": metadata,
        "tokenizer": tokenizer,
        "baseline_model": baseline_model,
        "stage_model": stage_model,
        "perm_vocab": perm_vocab,
        "inv_perm_vocab": inv_perm_vocab,
    }
