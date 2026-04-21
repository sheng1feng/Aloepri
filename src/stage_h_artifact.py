from __future__ import annotations

import json
from pathlib import Path

import torch

from src.defaults import DEFAULT_SEED
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_f import build_default_stage_f_keymat, build_layer_stage_f_configs, calibrate_keymat_kappas
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model


def build_stage_h_full_model_from_metadata(metadata: dict, recorder: TraceRecorder | None = None):
    model_dir = metadata["model_dir"]
    dtype = metadata.get("dtype", "float32")
    seed = int(metadata.get("seed", DEFAULT_SEED))
    lam = float(metadata["lambda"])
    h = int(metadata["h"])
    attention_profile = metadata["attention_profile"]
    adapted_layers = list(metadata["adapted_layers"])
    alpha_e = float(metadata.get("alpha_e", 0.0))
    alpha_h = float(metadata.get("alpha_h", 0.0))
    beta = int(metadata.get("beta", 8))
    gamma = float(metadata.get("gamma", 1e3))
    keymat_family = metadata.get("keymat_family", "algorithm1")

    set_global_seed(seed)
    tokenizer, baseline_model = load_model_and_tokenizer(model_dir, device="cpu", dtype=dtype)
    keymat_transform = build_default_stage_f_keymat(
        baseline_model,
        lam=lam,
        h=h,
        seed=seed,
        family=keymat_family,
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
        attention_profile=attention_profile,
        seed=seed,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
    )
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    stage_model, perm_vocab, inv_perm_vocab = build_stage_h_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        keymat_transform=keymat_transform,
        seed=seed,
        recorder=recorder,
        layer_configs=layer_configs_h,
        adapted_layers=adapted_layers,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        use_keymat_head=True,
        beta=beta,
        gamma=gamma,
    )
    return tokenizer, baseline_model, stage_model, perm_vocab, inv_perm_vocab


def save_stage_h_artifact(
    artifact_dir: str | Path,
    *,
    stage_model,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    metadata: dict,
) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    parameter_state = {name: tensor.detach().cpu() for name, tensor in stage_model.named_parameters()}
    buffer_state = {name: tensor.detach().cpu() for name, tensor in stage_model.named_buffers()}
    server_payload = {
        "parameter_state": parameter_state,
        "buffer_state": buffer_state,
    }
    client_secret = {
        "perm_vocab": perm_vocab.cpu(),
        "inv_perm_vocab": inv_perm_vocab.cpu(),
    }
    torch.save(
        {
            **server_payload,
            **client_secret,
        },
        artifact_dir / "model_state.pt",
    )
    torch.save(server_payload, artifact_dir / "server_model_state.pt")
    torch.save(client_secret, artifact_dir / "client_secret.pt")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stage_h_artifact(
    artifact_dir: str | Path,
    recorder: TraceRecorder | None = None,
):
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer, baseline_model, stage_model, perm_vocab, inv_perm_vocab = build_stage_h_full_model_from_metadata(
        metadata,
        recorder=recorder,
    )
    model_state_path = artifact_dir / "model_state.pt"
    server_state_path = artifact_dir / "server_model_state.pt"
    client_secret_path = artifact_dir / "client_secret.pt"
    if model_state_path.exists():
        payload = torch.load(model_state_path, map_location="cpu")
    else:
        payload = {
            **torch.load(server_state_path, map_location="cpu"),
            **torch.load(client_secret_path, map_location="cpu"),
        }
    parameter_map = dict(stage_model.named_parameters())
    buffer_map = dict(stage_model.named_buffers())
    for name, tensor in payload.get("parameter_state", {}).items():
        if name in parameter_map:
            parameter_map[name].data.copy_(tensor.to(device=parameter_map[name].device, dtype=parameter_map[name].dtype))
    for name, tensor in payload.get("buffer_state", {}).items():
        if name in buffer_map:
            buffer_map[name].data.copy_(tensor.to(device=buffer_map[name].device, dtype=buffer_map[name].dtype))
    return {
        "metadata": metadata,
        "tokenizer": tokenizer,
        "baseline_model": baseline_model,
        "stage_model": stage_model,
        "perm_vocab": payload["perm_vocab"],
        "inv_perm_vocab": payload["inv_perm_vocab"],
    }


def load_stage_h_server_artifact(
    artifact_dir: str | Path,
    recorder: TraceRecorder | None = None,
):
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer, baseline_model, stage_model, _, _ = build_stage_h_full_model_from_metadata(
        metadata,
        recorder=recorder,
    )
    payload = torch.load(artifact_dir / "server_model_state.pt", map_location="cpu")
    parameter_map = dict(stage_model.named_parameters())
    buffer_map = dict(stage_model.named_buffers())
    for name, tensor in payload.get("parameter_state", {}).items():
        if name in parameter_map:
            parameter_map[name].data.copy_(tensor.to(device=parameter_map[name].device, dtype=parameter_map[name].dtype))
    for name, tensor in payload.get("buffer_state", {}).items():
        if name in buffer_map:
            buffer_map[name].data.copy_(tensor.to(device=buffer_map[name].device, dtype=buffer_map[name].dtype))
    return {
        "metadata": metadata,
        "tokenizer": tokenizer,
        "baseline_model": baseline_model,
        "stage_model": stage_model,
    }
