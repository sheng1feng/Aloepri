from __future__ import annotations

import json
from pathlib import Path

import torch

from src.defaults import DEFAULT_SEED
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_f import build_default_stage_f_keymat, build_layer_stage_f_configs, calibrate_keymat_kappas
from src.stage_g import build_layer_stage_g_configs, build_stage_g_model
from src.stage_b import TraceRecorder


def build_stage_g_full_model_from_metadata(metadata: dict, recorder: TraceRecorder | None = None):
    model_dir = metadata["model_dir"]
    dtype = metadata.get("dtype", "float32")
    seed = int(metadata.get("seed", DEFAULT_SEED))
    lam = float(metadata["lambda"])
    h = int(metadata["h"])
    attention_profile = metadata["attention_profile"]
    adapted_layers = list(metadata["adapted_layers"])
    mode = metadata["mode"]
    alpha_e = float(metadata.get("alpha_e", 0.0))
    alpha_h = float(metadata.get("alpha_h", 0.0))

    set_global_seed(seed)
    tokenizer, baseline_model = load_model_and_tokenizer(model_dir, device="cpu", dtype=dtype)
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=lam, h=h, seed=seed)
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
    layer_configs_g = build_layer_stage_g_configs(layer_configs_f)
    stage_model, perm_vocab, inv_perm_vocab = build_stage_g_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        keymat_transform=keymat_transform,
        seed=seed,
        recorder=recorder,
        layer_configs=layer_configs_g,
        adapted_layers=adapted_layers,
        mode=mode,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        use_keymat_head=True,
    )
    return tokenizer, baseline_model, stage_model, perm_vocab, inv_perm_vocab


def save_stage_g_artifact(
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
    torch.save(
        {
            "parameter_state": parameter_state,
            "buffer_state": buffer_state,
            "perm_vocab": perm_vocab.cpu(),
            "inv_perm_vocab": inv_perm_vocab.cpu(),
        },
        artifact_dir / "model_state.pt",
    )
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stage_g_artifact(
    artifact_dir: str | Path,
    recorder: TraceRecorder | None = None,
):
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer, baseline_model, stage_model, perm_vocab, inv_perm_vocab = build_stage_g_full_model_from_metadata(
        metadata,
        recorder=recorder,
    )
    payload = torch.load(artifact_dir / "model_state.pt", map_location="cpu")
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
