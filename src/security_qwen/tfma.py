from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload
from src.security_qwen.artifacts import resolve_security_target


@dataclass(frozen=True)
class FrequencyAttackConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    knowledge_setting: str = "zero_knowledge"
    candidate_pool_size: int = 512
    topk: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_frequency_gate4_targets() -> list[str]:
    return [
        "stage_a_standard",
        "stage_h_full_obfuscated",
        "stage_j_stable_reference",
        "stage_j_tiny_a",
        "stage_k_stable_reference",
        "stage_k_tiny_a",
    ]


def build_tfma_template(
    target: SecurityEvalTarget,
    *,
    knowledge_setting: str = "zero_knowledge",
) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="tfma",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_frequency_attack_corpus",
            "seed": DEFAULT_SEED,
            "knowledge_setting": knowledge_setting,
        },
        metrics={
            "token_top10_recovery_rate": primary_metric,
            "token_top100_recovery_rate": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top10_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; TFMA implementation not started.",
        },
        artifacts={},
    )


def _generic_terms() -> dict[str, list[str]]:
    return {
        "subject": ["weather", "cooking", "travel", "sports", "music", "history"],
        "verb": ["explains", "summarizes", "discusses", "compares", "describes", "reviews"],
        "object": ["ideas", "tips", "examples", "concepts", "steps", "patterns"],
    }


def _privacy_terms() -> dict[str, list[str]]:
    return {
        "topic": ["privacy", "security", "token", "inference", "server", "client", "model", "prompt"],
        "verb": ["protects", "obfuscates", "hides", "restores", "analyzes", "evaluates"],
        "detail": ["weights", "tokens", "responses", "logs", "pipelines", "profiles"],
    }


def _expand_templates(templates: list[str], slots: dict[str, list[str]], limit: int) -> list[str]:
    keys = list(slots.keys())
    products = itertools.product(*(slots[key] for key in keys))
    texts: list[str] = []
    for combo in products:
        mapping = dict(zip(keys, combo))
        for template in templates:
            texts.append(template.format(**mapping))
            if len(texts) >= limit:
                return texts
    return texts


def build_frequency_corpora(knowledge_setting: str) -> dict[str, list[str]]:
    generic_templates = [
        "This {subject} note {verb} practical {object}.",
        "Please write about {subject} and provide {object}.",
        "A short article on {subject} {verb} useful {object}.",
    ]
    domain_templates = [
        "The {topic} system {verb} client {detail}.",
        "A report about {topic} {verb} server {detail}.",
        "This document on {topic} {verb} model {detail}.",
    ]
    private_templates = [
        "The private {topic} pipeline {verb} sensitive {detail}.",
        "A secure {topic} workflow {verb} hidden {detail}.",
        "This confidential {topic} system {verb} protected {detail}.",
    ]

    generic = _expand_templates(generic_templates, _generic_terms(), limit=180)
    domain = _expand_templates(domain_templates, _privacy_terms(), limit=180)
    private_a = _expand_templates(private_templates, _privacy_terms(), limit=180)
    private_b = list(reversed(private_a))

    if knowledge_setting == "zero_knowledge":
        return {"reference_texts": generic, "private_texts": private_a}
    if knowledge_setting == "domain_aware":
        return {"reference_texts": domain, "private_texts": private_a}
    if knowledge_setting == "distribution_aware":
        return {"reference_texts": private_b, "private_texts": private_a}
    raise ValueError(f"Unsupported knowledge_setting: {knowledge_setting}")


def load_frequency_bundle(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_dir, trust_remote_code=True)
    resolved = resolve_security_target(target_name)
    if resolved.client_secret_path is None:
        raise FileNotFoundError(f"Target {target_name} has no client secret")
    secret = torch.load(resolved.client_secret_path, map_location="cpu")
    return {
        "tokenizer": tokenizer,
        "perm_vocab": torch.as_tensor(secret["perm_vocab"], dtype=torch.long),
        "inv_perm_vocab": torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long),
        "resolved_target": resolved.to_dict(),
    }


def _tokenize_texts(tokenizer, texts: list[str]) -> list[list[int]]:
    sequences: list[list[int]] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            sequences.append(ids)
    return sequences


def _obfuscate_sequences(sequences: list[list[int]], perm_vocab: torch.Tensor) -> list[list[int]]:
    output: list[list[int]] = []
    for seq in sequences:
        tensor = torch.tensor(seq, dtype=torch.long)
        output.append(perm_vocab[tensor].tolist())
    return output


def _flatten_counts(sequences: list[list[int]]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for seq in sequences:
        counter.update(seq)
    return counter


def _collect_sensitive_plain_ids(tokenizer) -> set[int]:
    sensitive: set[int] = set()
    special = set(tokenizer.all_special_ids)
    for prompt in DEFAULT_PROMPTS:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        sensitive.update(int(item) for item in ids if item not in special)
    return sensitive


def _candidate_rank_lists(
    *,
    reference_counter: Counter[int],
    observed_counter: Counter[int],
    candidate_plain_ids: list[int],
    observed_token_ids: list[int],
) -> dict[int, list[int]]:
    plain_sorted = sorted(candidate_plain_ids, key=lambda token_id: (-reference_counter[token_id], token_id))
    plain_freq = torch.tensor([reference_counter[token_id] for token_id in plain_sorted], dtype=torch.float32)

    candidate_lists: dict[int, list[int]] = {}
    for observed_token in observed_token_ids:
        target_freq = float(observed_counter[observed_token])
        distances = torch.abs(plain_freq - target_freq)
        ranked_indices = torch.argsort(distances).tolist()
        candidate_lists[observed_token] = [plain_sorted[idx] for idx in ranked_indices]
    return candidate_lists


def _evaluate_rank_lists(
    *,
    candidate_lists: dict[int, list[int]],
    inv_perm_vocab: torch.Tensor,
    sensitive_plain_ids: set[int],
    topk_values: tuple[int, ...] = (1, 10, 100),
) -> dict[str, Any]:
    observed_token_ids = sorted(candidate_lists.keys())
    topk_hits = {k: [] for k in topk_values}
    sensitive_hits: list[bool] = []

    for observed_token in observed_token_ids:
        true_plain = int(inv_perm_vocab[observed_token].item())
        ranked = candidate_lists[observed_token]
        for k in topk_values:
            topk_hits[k].append(true_plain in ranked[: min(k, len(ranked))])
        if true_plain in sensitive_plain_ids:
            sensitive_hits.append(true_plain in ranked[:1])

    metrics = {
        "token_top1_recovery_rate": float(sum(topk_hits[1]) / max(len(topk_hits[1]), 1)),
        "token_top10_recovery_rate": float(sum(topk_hits[10]) / max(len(topk_hits[10]), 1)),
        "token_top100_recovery_rate": float(sum(topk_hits[100]) / max(len(topk_hits[100]), 1)),
        "sensitive_token_recovery_rate": float(sum(sensitive_hits) / len(sensitive_hits)) if sensitive_hits else None,
        "evaluated_obfuscated_token_count": len(observed_token_ids),
    }
    return metrics


def run_tfma_baseline(
    *,
    target_name: str,
    knowledge_setting: str = "zero_knowledge",
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    candidate_pool_size: int = 512,
    topk: int = 100,
) -> dict[str, Any]:
    bundle = load_frequency_bundle(target_name=target_name, baseline_model_dir=baseline_model_dir)
    tokenizer = bundle["tokenizer"]
    perm_vocab = bundle["perm_vocab"]
    inv_perm_vocab = bundle["inv_perm_vocab"]
    resolved = bundle["resolved_target"]
    corpora = build_frequency_corpora(knowledge_setting)
    reference_sequences = _tokenize_texts(tokenizer, corpora["reference_texts"])
    private_plain_sequences = _tokenize_texts(tokenizer, corpora["private_texts"])
    private_obfuscated_sequences = _obfuscate_sequences(private_plain_sequences, perm_vocab)

    reference_counter = _flatten_counts(reference_sequences)
    observed_counter = _flatten_counts(private_obfuscated_sequences)
    observed_token_ids = [token_id for token_id, _ in observed_counter.most_common(candidate_pool_size)]
    candidate_plain_ids = [token_id for token_id, _ in reference_counter.most_common(candidate_pool_size)]
    true_plain_ids = [int(inv_perm_vocab[token_id].item()) for token_id in observed_token_ids]
    candidate_plain_ids = list(dict.fromkeys(true_plain_ids + candidate_plain_ids))[:candidate_pool_size]

    candidate_lists = _candidate_rank_lists(
        reference_counter=reference_counter,
        observed_counter=observed_counter,
        candidate_plain_ids=candidate_plain_ids,
        observed_token_ids=observed_token_ids,
    )
    metrics = _evaluate_rank_lists(
        candidate_lists=candidate_lists,
        inv_perm_vocab=inv_perm_vocab,
        sensitive_plain_ids=_collect_sensitive_plain_ids(tokenizer),
        topk_values=(1, min(10, topk), min(100, topk)),
    )
    payload = build_security_eval_payload(
        attack="tfma",
        target=SecurityEvalTarget(
            stage=resolved["stage"],
            artifact_dir=resolved["artifact_dir"],
            profile=resolved["profile"],
            model_family="qwen",
            variant=resolved["variant"],
        ),
        config={
            "phase": "gate4_minimal",
            **FrequencyAttackConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                knowledge_setting=knowledge_setting,
                candidate_pool_size=candidate_pool_size,
                topk=topk,
            ).to_dict(),
            "dataset_name": "phase4_local_frequency_corpora",
        },
        metrics={
            **metrics,
            "reference_sequence_count": len(reference_sequences),
            "private_sequence_count": len(private_plain_sequences),
        },
        summary={
            "status": "completed_minimal_baseline",
            "primary_metric_name": "token_top10_recovery_rate",
            "primary_metric_value": metrics["token_top10_recovery_rate"],
            "risk_level": classify_risk_level(metrics["token_top10_recovery_rate"]),
            "notes": "Gate-4 minimal TFMA using unigram frequency matching on local reproducible corpora.",
        },
        artifacts={
            "resolved_target": resolved,
        },
    )
    return payload


def build_tfma_comparison_payload(
    *,
    result_payloads: list[dict[str, Any]],
    baseline_target_name: str = "stage_a_standard",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_by_setting: dict[str, float] = {}
    for payload in result_payloads:
        target = payload["target"]
        metrics = payload["metrics"]
        config = payload["config"]
        summary = payload["summary"]
        row = {
            "stage": target.get("stage"),
            "profile": target.get("profile"),
            "artifact_dir": target.get("artifact_dir"),
            "variant": target.get("variant"),
            "knowledge_setting": config.get("knowledge_setting"),
            "token_top1_recovery_rate": metrics.get("token_top1_recovery_rate"),
            "token_top10_recovery_rate": metrics.get("token_top10_recovery_rate"),
            "token_top100_recovery_rate": metrics.get("token_top100_recovery_rate"),
            "sensitive_token_recovery_rate": metrics.get("sensitive_token_recovery_rate"),
            "risk_level": summary.get("risk_level"),
            "status": summary.get("status"),
            "target_name": payload.get("artifacts", {}).get("resolved_target", {}).get("name"),
        }
        rows.append(row)
        if row["target_name"] == baseline_target_name:
            baseline_by_setting[row["knowledge_setting"]] = row["token_top10_recovery_rate"]

    for row in rows:
        base = baseline_by_setting.get(row["knowledge_setting"])
        value = row["token_top10_recovery_rate"]
        row["vs_stage_a_top10_delta"] = None if base is None or value is None else float(value - base)

    rows.sort(key=lambda item: (item["knowledge_setting"], -(item["token_top10_recovery_rate"] or -1.0), item["stage"] or ""))
    return {
        "format": "qwen_security_tfma_comparison_v1",
        "baseline_target_name": baseline_target_name,
        "row_count": len(rows),
        "rows": rows,
    }
