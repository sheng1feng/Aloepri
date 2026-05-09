from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.model_loader import set_global_seed
from src.security_qwen.artifacts import resolve_security_target
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


@dataclass(frozen=True)
class IMABaselineConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    train_size: int = 1024
    val_size: int = 128
    test_size: int = 128
    candidate_pool_size: int = 2048
    topk: int = 10
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ridge_alphas"] = list(self.ridge_alphas)
        return payload


DEFAULT_PAPER_LIKE_CORPUS_PATHS: tuple[str, ...] = (
    "docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).txt",
    "docs/AloePri 论文中的部署适配机制整理.md",
    "docs/AloePri_技术报告梳理与复现方案.md",
    "README.md",
)


@dataclass(frozen=True)
class IMAPaperLikeConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    sequence_length: int = 32
    train_sequence_count: int = 128
    val_sequence_count: int = 16
    test_sequence_count: int = 16
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    topk: int = 10
    candidate_pool_strategy: str = "full_movable_vocab"
    public_corpus_paths: tuple[str, ...] = DEFAULT_PAPER_LIKE_CORPUS_PATHS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["public_corpus_paths"] = list(self.public_corpus_paths)
        return payload


def default_ima_gate2_targets() -> list[str]:
    return [
        "stage_a_standard",
        "stage_h_full_obfuscated",
        "stage_j_stable_reference",
        "stage_j_tiny_a",
        "stage_k_reference",
        "stage_k_default",
    ]


def build_ima_template(target: SecurityEvalTarget) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="ima",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_inversion_public_corpus",
            "seed": DEFAULT_SEED,
            "prediction_targets": ["token_id", "embedding"],
        },
        metrics={
            "token_top1_recovery_rate": primary_metric,
            "token_top10_recovery_rate": None,
            "embedding_cosine_similarity": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; IMA implementation not started.",
        },
        artifacts={},
    )


def default_ima_output_path(target_name: str, *, mode: str = "minimal") -> Path:
    suffix = ".paper_like.json" if mode == "paper_like" else ".json"
    return Path(f"outputs/security_qwen/ima/{target_name}{suffix}")


def _load_public_inversion_texts(corpus_paths: Iterable[str]) -> list[str]:
    texts: list[str] = []
    for item in corpus_paths:
        path = Path(item)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
    return texts


def build_paper_like_inverter_config(
    *,
    observed_hidden_size: int,
    vocab_size: int,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
):
    if observed_hidden_size % 8 != 0:
        raise ValueError(f"observed_hidden_size must be divisible by 8, got {observed_hidden_size}")
    config = AutoConfig.from_pretrained(baseline_model_dir, trust_remote_code=True)
    config.hidden_size = int(observed_hidden_size)
    config.num_hidden_layers = 2
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.intermediate_size = max(int(observed_hidden_size) * 4, int(getattr(config, "intermediate_size", 0) or 0))
    config.vocab_size = int(vocab_size)
    config.torch_dtype = "float32"
    return config


class _PaperLikeIMAInverter(nn.Module):
    def __init__(self, *, backbone_config, target_embedding_dim: int) -> None:
        super().__init__()
        self.backbone = AutoModel.from_config(backbone_config)
        self.output_proj = nn.Linear(int(backbone_config.hidden_size), int(target_embedding_dim), bias=False)
        self.float()

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
        return self.output_proj(hidden)


def _resolve_ima_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _collect_public_token_windows(
    *,
    tokenizer,
    corpus_paths: Iterable[str],
    sequence_length: int,
    train_sequence_count: int,
    val_sequence_count: int,
    test_sequence_count: int,
    seed: int,
) -> dict[str, Any]:
    texts = _load_public_inversion_texts(corpus_paths)
    if not texts:
        raise FileNotFoundError("No readable public inversion corpus texts were found.")

    all_tokens: list[int] = []
    for text in texts:
        all_tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    all_tokens = [int(token_id) for token_id in all_tokens if 0 <= int(token_id) < tokenizer.vocab_size]

    windows: list[list[int]] = []
    step = int(sequence_length)
    for start in range(0, max(len(all_tokens) - step + 1, 0), step):
        window = all_tokens[start : start + step]
        if len(window) == step:
            windows.append(window)

    required = int(train_sequence_count) + int(val_sequence_count) + int(test_sequence_count)
    if len(windows) < required:
        raise ValueError(f"Need at least {required} public windows, found {len(windows)}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(windows), generator=generator).tolist()
    shuffled = [windows[index] for index in order[:required]]

    train_end = int(train_sequence_count)
    val_end = train_end + int(val_sequence_count)
    train = torch.tensor(shuffled[:train_end], dtype=torch.long)
    val = torch.tensor(shuffled[train_end:val_end], dtype=torch.long)
    test = torch.tensor(shuffled[val_end:required], dtype=torch.long)
    return {
        "texts": texts,
        "train_plain_ids": train,
        "val_plain_ids": val,
        "test_plain_ids": test,
    }


def _rank_candidate_plain_ids(
    *,
    predicted_embeddings: torch.Tensor,
    candidate_plain_ids: torch.Tensor,
    baseline_embed: torch.Tensor,
    topk: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    pred_norm = predicted_embeddings / predicted_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    best_scores = torch.full((pred_norm.shape[0], 0), float("-inf"), dtype=pred_norm.dtype, device=pred_norm.device)
    best_plain_ids = torch.empty((pred_norm.shape[0], 0), dtype=torch.long, device=pred_norm.device)
    k = min(int(topk), int(candidate_plain_ids.numel()))

    for start in range(0, int(candidate_plain_ids.numel()), int(chunk_size)):
        end = min(start + int(chunk_size), int(candidate_plain_ids.numel()))
        chunk_ids = candidate_plain_ids[start:end]
        chunk_embeddings = baseline_embed[chunk_ids]
        chunk_norm = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scores = pred_norm @ chunk_norm.T
        chunk_scores, chunk_indices = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
        chunk_plain_ids = chunk_ids[chunk_indices]

        merged_scores = torch.cat([best_scores, chunk_scores], dim=1)
        merged_plain_ids = torch.cat([best_plain_ids, chunk_plain_ids], dim=1)
        new_scores, new_indices = torch.topk(merged_scores, k=k, dim=1)
        best_scores = new_scores
        best_plain_ids = merged_plain_ids.gather(1, new_indices)

    return best_plain_ids


def _evaluate_sequence_inversion_predictions(
    *,
    predicted_embeddings: torch.Tensor,
    true_plain_ids: torch.Tensor,
    candidate_plain_ids: torch.Tensor,
    baseline_embed: torch.Tensor,
    sensitive_plain_ids: torch.Tensor,
    topk: int,
) -> dict[str, Any]:
    flat_pred = predicted_embeddings.reshape(-1, predicted_embeddings.shape[-1])
    flat_true = true_plain_ids.reshape(-1)
    predicted_plain_ids = _rank_candidate_plain_ids(
        predicted_embeddings=flat_pred,
        candidate_plain_ids=candidate_plain_ids,
        baseline_embed=baseline_embed,
        topk=topk,
    )
    hits = predicted_plain_ids.eq(flat_true.unsqueeze(1))
    true_embeddings = baseline_embed[flat_true]
    top1_embeddings = baseline_embed[predicted_plain_ids[:, 0]]
    cosine = _row_cosine_similarity(top1_embeddings, true_embeddings)

    sensitive_rate = None
    sensitive_mask = torch.isin(flat_true, sensitive_plain_ids) if sensitive_plain_ids.numel() > 0 else torch.zeros_like(flat_true, dtype=torch.bool)
    if bool(sensitive_mask.any()):
        sensitive_rate = float(
            predicted_plain_ids[sensitive_mask, 0].eq(flat_true[sensitive_mask]).to(torch.float32).mean().item()
        )

    return {
        "token_top1_recovery_rate": float(hits[:, 0].to(torch.float32).mean().item()),
        "token_top10_recovery_rate": float(hits[:, : min(10, hits.shape[1])].any(dim=1).to(torch.float32).mean().item()),
        "embedding_cosine_similarity": float(cosine.mean().item()),
        "sensitive_token_recovery_rate": sensitive_rate,
        "predicted_plain_ids_sample": predicted_plain_ids[:10, 0].tolist(),
        "evaluated_token_count": int(flat_true.numel()),
    }


def _load_standard_embedding(server_dir: str) -> torch.Tensor:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    return state["model.embed_tokens.weight"].to(torch.float32)


def _load_buffered_stage_embedding(server_dir: str) -> torch.Tensor:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    return state["buffer::stage_a_model.model.embed_tokens.weight"].to(torch.float32)


def _load_stage_h_embedding(resolved_root_dir: str) -> torch.Tensor:
    payload = torch.load(Path(resolved_root_dir) / "server_model_state.pt", map_location="cpu")
    buffers = payload.get("buffer_state", {})
    return torch.as_tensor(buffers["stage_a_model.model.embed_tokens.weight"], dtype=torch.float32)


def load_ima_embedding_sources(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    baseline_state = load_file(str(Path(baseline_model_dir) / "model.safetensors"))
    baseline_embed = baseline_state["model.embed_tokens.weight"].to(torch.float32)

    resolved = resolve_security_target(target_name)
    if resolved.stage == "H":
        observed_embed = _load_stage_h_embedding(resolved.resolved_root_dir)
    else:
        if resolved.server_dir is None:
            raise FileNotFoundError(f"Target {target_name} has no server_dir")
        state = load_file(str(Path(resolved.server_dir) / "model.safetensors"))
        if "buffer::stage_a_model.model.embed_tokens.weight" in state:
            observed_embed = state["buffer::stage_a_model.model.embed_tokens.weight"].to(torch.float32)
        else:
            observed_embed = state["model.embed_tokens.weight"].to(torch.float32)

    if resolved.client_secret_path is None:
        raise FileNotFoundError(f"Target {target_name} has no client_secret_path")
    secret = torch.load(resolved.client_secret_path, map_location="cpu")
    metadata = {}
    if resolved.metadata_path is not None and Path(resolved.metadata_path).exists():
        metadata = json.loads(Path(resolved.metadata_path).read_text(encoding="utf-8"))
    return baseline_embed, observed_embed, {
        "resolved_target": resolved.to_dict(),
        "perm_vocab": torch.as_tensor(secret["perm_vocab"], dtype=torch.long),
        "inv_perm_vocab": torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long),
        "metadata": metadata,
    }


def _collect_sensitive_plain_ids(tokenizer) -> torch.Tensor:
    sensitive: set[int] = set()
    special_ids = set(tokenizer.all_special_ids)
    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for token_id in input_ids:
            if token_id not in special_ids and token_id < tokenizer.vocab_size:
                sensitive.add(int(token_id))
    return torch.tensor(sorted(sensitive), dtype=torch.long)


def _unique_preserve_order(items: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _sample_ima_splits(
    *,
    tokenizer,
    train_size: int,
    val_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
    sensitive_plain_ids_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    movable_plain_ids = ordinary_token_ids(tokenizer).tolist()
    sensitive_source = sensitive_plain_ids_override if sensitive_plain_ids_override is not None else _collect_sensitive_plain_ids(tokenizer)
    sensitive_plain_ids = [item for item in torch.as_tensor(sensitive_source, dtype=torch.long).tolist() if item in set(movable_plain_ids)]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled = torch.tensor(movable_plain_ids, dtype=torch.long)[torch.randperm(len(movable_plain_ids), generator=generator)].tolist()
    remainder = [item for item in shuffled if item not in sensitive_plain_ids]

    test_plain = _unique_preserve_order(sensitive_plain_ids + remainder[: max(test_size - len(sensitive_plain_ids), 0)])
    remainder = [item for item in remainder if item not in set(test_plain)]

    val_plain = remainder[:val_size]
    remainder = remainder[val_size:]
    train_plain = remainder[:train_size]
    remainder = remainder[train_size:]

    candidate_plain = _unique_preserve_order(test_plain + remainder[: max(candidate_pool_size - len(test_plain), 0)])

    return {
        "train_plain_ids": torch.tensor(train_plain, dtype=torch.long),
        "val_plain_ids": torch.tensor(val_plain, dtype=torch.long),
        "test_plain_ids": torch.tensor(test_plain, dtype=torch.long),
        "candidate_plain_ids": torch.tensor(candidate_plain, dtype=torch.long),
        "sensitive_plain_ids": torch.tensor(sensitive_plain_ids, dtype=torch.long),
    }


def _fit_ridge_regressor(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    ridge_alpha: float,
) -> dict[str, torch.Tensor]:
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_norm = (x_train - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std
    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)
    x_aug = torch.cat([x_norm, ones], dim=1)
    dim = x_aug.shape[1]
    identity = torch.eye(dim, dtype=x_norm.dtype)
    identity[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + ridge_alpha * identity
    rhs = x_aug.T @ y_norm
    weight = torch.linalg.solve(lhs, rhs)
    return {
        "weight": weight,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _predict_ridge(model: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x_norm = (x - model["x_mean"]) / model["x_std"]
    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)
    x_aug = torch.cat([x_norm, ones], dim=1)
    y_norm = x_aug @ model["weight"]
    return y_norm * model["y_std"] + model["y_mean"]


def _row_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    y_norm = y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (x_norm * y_norm).sum(dim=1)


def _evaluate_inversion_predictions(
    *,
    predicted_embeddings: torch.Tensor,
    true_plain_ids: torch.Tensor,
    candidate_plain_ids: torch.Tensor,
    baseline_embed: torch.Tensor,
    topk: int,
) -> dict[str, Any]:
    candidate_embeddings = baseline_embed[candidate_plain_ids]
    pred_norm = predicted_embeddings / predicted_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cand_norm = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    score_matrix = pred_norm @ cand_norm.T
    k = min(topk, score_matrix.shape[1])
    topk_indices = torch.topk(score_matrix, k=k, dim=1).indices
    predicted_plain_ids = candidate_plain_ids[topk_indices]
    hits = predicted_plain_ids.eq(true_plain_ids.unsqueeze(1))
    true_embeddings = baseline_embed[true_plain_ids]
    cosine = _row_cosine_similarity(predicted_embeddings, true_embeddings)
    return {
        "token_top1_recovery_rate": float(hits[:, 0].to(torch.float32).mean().item()),
        "token_top10_recovery_rate": float(hits[:, : min(10, hits.shape[1])].any(dim=1).to(torch.float32).mean().item()),
        "embedding_cosine_similarity": float(cosine.mean().item()),
        "predicted_plain_ids_sample": predicted_plain_ids[:10, 0].tolist(),
    }


def run_ima_baseline(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    train_size: int = 1024,
    val_size: int = 128,
    test_size: int = 128,
    candidate_pool_size: int = 2048,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    sensitive_plain_ids_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    baseline_embed, observed_embed, aux = load_ima_embedding_sources(
        target_name=target_name,
        baseline_model_dir=baseline_model_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_dir, trust_remote_code=True)
    splits = _sample_ima_splits(
        tokenizer=tokenizer,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        sensitive_plain_ids_override=sensitive_plain_ids_override,
    )
    perm_vocab = aux["perm_vocab"]

    train_plain_ids = splits["train_plain_ids"]
    val_plain_ids = splits["val_plain_ids"]
    test_plain_ids = splits["test_plain_ids"]
    candidate_plain_ids = splits["candidate_plain_ids"]
    sensitive_plain_ids = splits["sensitive_plain_ids"]

    x_train = observed_embed[perm_vocab[train_plain_ids]]
    y_train = baseline_embed[train_plain_ids]
    x_val = observed_embed[perm_vocab[val_plain_ids]]
    y_val = baseline_embed[val_plain_ids]
    x_test = observed_embed[perm_vocab[test_plain_ids]]

    val_candidate_plain_ids = torch.unique(torch.cat([val_plain_ids, candidate_plain_ids], dim=0), sorted=False)

    best_alpha = None
    best_val_top1 = -1.0
    best_model = None
    alpha_scores: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        model = _fit_ridge_regressor(x_train, y_train, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model, x_val)
        val_metrics = _evaluate_inversion_predictions(
            predicted_embeddings=val_pred,
            true_plain_ids=val_plain_ids,
            candidate_plain_ids=val_candidate_plain_ids,
            baseline_embed=baseline_embed,
            topk=topk,
        )
        alpha_scores.append({"ridge_alpha": float(alpha), **val_metrics})
        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_val_top1 = val_metrics["token_top1_recovery_rate"]
            best_alpha = float(alpha)
            best_model = model

    if best_model is None or best_alpha is None:
        raise RuntimeError("IMA failed to select a best ridge model")

    test_pred = _predict_ridge(best_model, x_test)
    test_metrics = _evaluate_inversion_predictions(
        predicted_embeddings=test_pred,
        true_plain_ids=test_plain_ids,
        candidate_plain_ids=candidate_plain_ids,
        baseline_embed=baseline_embed,
        topk=topk,
    )

    sensitive_mask = torch.isin(test_plain_ids, sensitive_plain_ids) if sensitive_plain_ids.numel() > 0 else torch.zeros_like(test_plain_ids, dtype=torch.bool)
    sensitive_rate = None
    if bool(sensitive_mask.any()):
        candidate_embeddings = baseline_embed[candidate_plain_ids]
        pred_norm = test_pred / test_pred.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cand_norm = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        score_matrix = pred_norm @ cand_norm.T
        top1_indices = torch.topk(score_matrix, k=1, dim=1).indices.squeeze(1)
        predicted_plain = candidate_plain_ids[top1_indices]
        sensitive_rate = float(predicted_plain[sensitive_mask].eq(test_plain_ids[sensitive_mask]).to(torch.float32).mean().item())

    runtime_seconds = time.perf_counter() - start_time
    resolved = aux["resolved_target"]
    payload = build_security_eval_payload(
        attack="ima",
        target=SecurityEvalTarget(
            stage=resolved["stage"],
            artifact_dir=resolved["artifact_dir"],
            profile=resolved["profile"],
            model_family="qwen",
            variant=resolved["variant"],
        ),
        config={
            "phase": "gate2_minimal",
            **IMABaselineConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                train_size=int(train_plain_ids.numel()),
                val_size=int(val_plain_ids.numel()),
                test_size=int(test_plain_ids.numel()),
                candidate_pool_size=int(candidate_plain_ids.numel()),
                topk=topk,
                ridge_alphas=tuple(float(item) for item in ridge_alphas),
            ).to_dict(),
            "dataset_name": "phase0_token_row_proxy_corpus",
            "prediction_targets": ["embedding", "token_id"],
            "selected_ridge_alpha": best_alpha,
            "alpha_scores": alpha_scores,
        },
        metrics={
            **test_metrics,
            "sensitive_token_recovery_rate": sensitive_rate,
            "train_size": int(train_plain_ids.numel()),
            "val_size": int(val_plain_ids.numel()),
            "test_size": int(test_plain_ids.numel()),
            "candidate_plain_token_count": int(candidate_plain_ids.numel()),
            "attack_runtime_seconds": runtime_seconds,
        },
        summary={
            "status": "completed_minimal_baseline",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": test_metrics["token_top1_recovery_rate"],
            "risk_level": classify_risk_level(test_metrics["token_top1_recovery_rate"]),
            "notes": "Gate-2 minimal ridge inversion baseline on obfuscated embedding rows.",
        },
        artifacts={
            "resolved_target": resolved,
            "predicted_plain_ids_sample": test_metrics["predicted_plain_ids_sample"],
        },
    )
    return payload


def run_ima_paper_like(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    sequence_length: int = 32,
    train_sequence_count: int = 128,
    val_sequence_count: int = 16,
    test_sequence_count: int = 16,
    batch_size: int = 8,
    epochs: int = 2,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    topk: int = 10,
    device: str = "auto",
    public_corpus_paths: Iterable[str] = DEFAULT_PAPER_LIKE_CORPUS_PATHS,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    set_global_seed(seed)
    resolved_corpus_paths = tuple(str(item) for item in public_corpus_paths) or DEFAULT_PAPER_LIKE_CORPUS_PATHS

    baseline_embed, observed_embed, aux = load_ima_embedding_sources(
        target_name=target_name,
        baseline_model_dir=baseline_model_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_dir, trust_remote_code=True)
    corpus = _collect_public_token_windows(
        tokenizer=tokenizer,
        corpus_paths=resolved_corpus_paths,
        sequence_length=sequence_length,
        train_sequence_count=train_sequence_count,
        val_sequence_count=val_sequence_count,
        test_sequence_count=test_sequence_count,
        seed=seed,
    )
    sensitive_plain_ids = _collect_sensitive_plain_ids(tokenizer)
    candidate_plain_ids = ordinary_token_ids(tokenizer)
    perm_vocab = aux["perm_vocab"]

    train_plain_ids = corpus["train_plain_ids"]
    val_plain_ids = corpus["val_plain_ids"]
    test_plain_ids = corpus["test_plain_ids"]

    x_train = observed_embed[perm_vocab[train_plain_ids]]
    y_train = baseline_embed[train_plain_ids]
    x_val = observed_embed[perm_vocab[val_plain_ids]]
    x_test = observed_embed[perm_vocab[test_plain_ids]]

    resolved_device = _resolve_ima_device(device)
    attack_config = build_paper_like_inverter_config(
        observed_hidden_size=int(observed_embed.shape[1]),
        vocab_size=int(tokenizer.vocab_size),
        baseline_model_dir=baseline_model_dir,
    )
    model = _PaperLikeIMAInverter(
        backbone_config=attack_config,
        target_embedding_dim=int(baseline_embed.shape[1]),
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    x_val_device = x_val.to(resolved_device)
    x_test_device = x_test.to(resolved_device)
    baseline_embed_device = baseline_embed.to(resolved_device)
    candidate_plain_ids_device = candidate_plain_ids.to(resolved_device)
    sensitive_plain_ids_device = sensitive_plain_ids.to(resolved_device)

    best_epoch = -1
    best_val_top1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epoch_summaries: list[dict[str, Any]] = []

    for epoch_idx in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_inputs)
            loss = torch.nn.functional.mse_loss(pred, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_device)
            val_metrics = _evaluate_sequence_inversion_predictions(
                predicted_embeddings=val_pred,
                true_plain_ids=val_plain_ids.to(resolved_device),
                candidate_plain_ids=candidate_plain_ids_device,
                baseline_embed=baseline_embed_device,
                sensitive_plain_ids=sensitive_plain_ids_device,
                topk=topk,
            )

        epoch_summary = {
            "epoch": epoch_idx + 1,
            "train_loss": total_loss / max(total_batches, 1),
            "val_token_top1_recovery_rate": val_metrics["token_top1_recovery_rate"],
            "val_token_top10_recovery_rate": val_metrics["token_top10_recovery_rate"],
            "val_embedding_cosine_similarity": val_metrics["embedding_cosine_similarity"],
        }
        epoch_summaries.append(epoch_summary)

        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_val_top1 = float(val_metrics["token_top1_recovery_rate"])
            best_epoch = epoch_idx + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("IMA paper-like attack failed to produce a best checkpoint")

    model.load_state_dict(best_state)
    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_device)
        test_metrics = _evaluate_sequence_inversion_predictions(
            predicted_embeddings=test_pred,
            true_plain_ids=test_plain_ids.to(resolved_device),
            candidate_plain_ids=candidate_plain_ids_device,
            baseline_embed=baseline_embed_device,
            sensitive_plain_ids=sensitive_plain_ids_device,
            topk=topk,
        )

    runtime_seconds = time.perf_counter() - start_time
    resolved = aux["resolved_target"]
    payload = build_security_eval_payload(
        attack="ima",
        target=SecurityEvalTarget(
            stage=resolved["stage"],
            artifact_dir=resolved["artifact_dir"],
            profile=resolved["profile"],
            model_family="qwen",
            variant=resolved["variant"],
        ),
        config={
            "phase": "paper_like",
            **IMAPaperLikeConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                sequence_length=sequence_length,
                train_sequence_count=int(train_plain_ids.shape[0]),
                val_sequence_count=int(val_plain_ids.shape[0]),
                test_sequence_count=int(test_plain_ids.shape[0]),
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                topk=topk,
                public_corpus_paths=resolved_corpus_paths,
            ).to_dict(),
            "dataset_name": "phase0_inversion_public_corpus_local_docs",
            "prediction_targets": ["embedding", "token_id"],
            "attack_model_family": "qwen2",
            "attack_hidden_size": int(attack_config.hidden_size),
            "attack_num_hidden_layers": int(attack_config.num_hidden_layers),
            "attack_num_attention_heads": int(attack_config.num_attention_heads),
            "device": resolved_device,
            "best_epoch": best_epoch,
            "epoch_summaries": epoch_summaries,
        },
        metrics={
            **test_metrics,
            "train_sequence_count": int(train_plain_ids.shape[0]),
            "val_sequence_count": int(val_plain_ids.shape[0]),
            "test_sequence_count": int(test_plain_ids.shape[0]),
            "sequence_length": int(sequence_length),
            "candidate_plain_token_count": int(candidate_plain_ids.numel()),
            "attack_runtime_seconds": runtime_seconds,
        },
        summary={
            "status": "completed_paper_like",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": test_metrics["token_top1_recovery_rate"],
            "risk_level": classify_risk_level(test_metrics["token_top1_recovery_rate"]),
            "notes": "Paper-like Qwen2 embedding inversion attack trained on local public docs with a 2-layer, 8-head inverter.",
        },
        artifacts={
            "resolved_target": resolved,
            "predicted_plain_ids_sample": test_metrics["predicted_plain_ids_sample"],
            "public_corpus_paths": list(resolved_corpus_paths),
            "public_corpus_document_count": len(corpus["texts"]),
        },
    )
    return payload


def build_ima_comparison_payload(
    *,
    result_payloads: list[dict[str, Any]],
    baseline_target_name: str = "stage_a_standard",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] | None = None

    for payload in result_payloads:
        target = payload["target"]
        metrics = payload["metrics"]
        summary = payload["summary"]
        row = {
            "stage": target.get("stage"),
            "profile": target.get("profile"),
            "artifact_dir": target.get("artifact_dir"),
            "variant": target.get("variant"),
            "phase": payload.get("config", {}).get("phase"),
            "token_top1_recovery_rate": metrics.get("token_top1_recovery_rate"),
            "token_top10_recovery_rate": metrics.get("token_top10_recovery_rate"),
            "embedding_cosine_similarity": metrics.get("embedding_cosine_similarity"),
            "sensitive_token_recovery_rate": metrics.get("sensitive_token_recovery_rate"),
            "risk_level": summary.get("risk_level"),
            "status": summary.get("status"),
            "selected_ridge_alpha": payload.get("config", {}).get("selected_ridge_alpha"),
            "target_name": payload.get("artifacts", {}).get("resolved_target", {}).get("name"),
        }
        rows.append(row)
        if row["target_name"] == baseline_target_name:
            baseline_row = row

    baseline_top1 = baseline_row.get("token_top1_recovery_rate") if baseline_row is not None else None
    baseline_cos = baseline_row.get("embedding_cosine_similarity") if baseline_row is not None else None
    for row in rows:
        row["vs_stage_a_top1_delta"] = None if baseline_top1 is None or row["token_top1_recovery_rate"] is None else float(row["token_top1_recovery_rate"] - baseline_top1)
        row["vs_stage_a_cosine_delta"] = None if baseline_cos is None or row["embedding_cosine_similarity"] is None else float(row["embedding_cosine_similarity"] - baseline_cos)

    rows.sort(key=lambda item: (item["token_top1_recovery_rate"] is None, -(item["token_top1_recovery_rate"] or -1.0)))
    return {
        "format": "qwen_security_ima_comparison_v1",
        "baseline_target_name": baseline_target_name,
        "row_count": len(rows),
        "rows": rows,
    }
