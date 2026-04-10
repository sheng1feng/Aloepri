from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import torch

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload
from src.security_qwen.tfma import (
    _collect_sensitive_plain_ids,
    _evaluate_rank_lists,
    _flatten_counts,
    _obfuscate_sequences,
    _tokenize_texts,
    build_frequency_corpora,
    default_frequency_gate4_targets,
    load_frequency_bundle,
)


@dataclass(frozen=True)
class SDABaselineConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    knowledge_setting: str = "zero_knowledge"
    candidate_pool_size: int = 256
    topk: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sda_template(
    target: SecurityEvalTarget,
    *,
    knowledge_setting: str = "zero_knowledge",
) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="sda",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_frequency_attack_corpus",
            "seed": DEFAULT_SEED,
            "knowledge_setting": knowledge_setting,
        },
        metrics={
            "bleu4": primary_metric,
            "token_top100_recovery_rate": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "bleu4",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; SDA implementation not started.",
        },
        artifacts={},
    )


def _build_bigram_matrix(sequences: list[list[int]], active_token_ids: list[int]) -> torch.Tensor:
    index = {token_id: idx for idx, token_id in enumerate(active_token_ids)}
    matrix = torch.zeros((len(active_token_ids), len(active_token_ids)), dtype=torch.float32)
    for seq in sequences:
        for left, right in zip(seq[:-1], seq[1:]):
            if left in index and right in index:
                matrix[index[left], index[right]] += 1.0
    return matrix


def _sorted_bigram_signature(
    matrix: torch.Tensor,
    unigram_counter: Counter[int],
    token_ids: list[int],
    *,
    target_width: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for idx, token_id in enumerate(token_ids):
        outgoing = torch.sort(matrix[idx]).values
        incoming = torch.sort(matrix[:, idx]).values
        if outgoing.numel() < target_width:
            outgoing = torch.nn.functional.pad(outgoing, (0, target_width - outgoing.numel()))
        if incoming.numel() < target_width:
            incoming = torch.nn.functional.pad(incoming, (0, target_width - incoming.numel()))
        freq = torch.tensor([float(unigram_counter[token_id])], dtype=torch.float32)
        feature = torch.cat([freq, outgoing, incoming], dim=0)
        feature = feature - feature.mean()
        feature = feature / feature.norm().clamp_min(1e-8)
        rows.append(feature)
    return torch.stack(rows, dim=0)


def _build_candidate_rank_lists(
    *,
    reference_sequences: list[list[int]],
    private_obfuscated_sequences: list[list[int]],
    candidate_plain_ids: list[int],
    observed_token_ids: list[int],
) -> dict[int, list[int]]:
    reference_counter = _flatten_counts(reference_sequences)
    observed_counter = _flatten_counts(private_obfuscated_sequences)
    ref_matrix = _build_bigram_matrix(reference_sequences, candidate_plain_ids)
    obs_matrix = _build_bigram_matrix(private_obfuscated_sequences, observed_token_ids)
    target_width = max(len(candidate_plain_ids), len(observed_token_ids))
    ref_features = _sorted_bigram_signature(ref_matrix, reference_counter, candidate_plain_ids, target_width=target_width)
    obs_features = _sorted_bigram_signature(obs_matrix, observed_counter, observed_token_ids, target_width=target_width)
    score_matrix = obs_features @ ref_features.T
    rank_lists: dict[int, list[int]] = {}
    for row_idx, observed_token in enumerate(observed_token_ids):
        ranked = torch.argsort(score_matrix[row_idx], descending=True).tolist()
        rank_lists[observed_token] = [candidate_plain_ids[idx] for idx in ranked]
    return rank_lists


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _token_bleu4(reference_sequences: list[list[int]], hypothesis_sequences: list[list[int]]) -> float:
    ref_tokens = [str(token) for seq in reference_sequences for token in seq]
    hyp_tokens = [str(token) for seq in hypothesis_sequences for token in seq]
    if not ref_tokens or not hyp_tokens:
        return 0.0
    precisions: list[float] = []
    for n in range(1, 5):
        ref_counts = _ngrams(ref_tokens, n)
        hyp_counts = _ngrams(hyp_tokens, n)
        overlap = sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
        total = max(sum(hyp_counts.values()), 1)
        precisions.append(max(overlap / total, 1e-8))
    import math

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    brevity_penalty = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return float(brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0))


def run_sda_baseline(
    *,
    target_name: str,
    knowledge_setting: str = "zero_knowledge",
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    candidate_pool_size: int = 256,
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

    rank_lists = _build_candidate_rank_lists(
        reference_sequences=reference_sequences,
        private_obfuscated_sequences=private_obfuscated_sequences,
        candidate_plain_ids=candidate_plain_ids,
        observed_token_ids=observed_token_ids,
    )
    metrics = _evaluate_rank_lists(
        candidate_lists=rank_lists,
        inv_perm_vocab=inv_perm_vocab,
        sensitive_plain_ids=_collect_sensitive_plain_ids(tokenizer),
        topk_values=(1, min(10, topk), min(100, topk)),
    )

    best_plain_map = {observed_token: ranked[0] for observed_token, ranked in rank_lists.items()}
    reconstructed_sequences = [
        [best_plain_map.get(token_id, int(inv_perm_vocab[token_id].item())) for token_id in seq]
        for seq in private_obfuscated_sequences
    ]
    bleu4 = _token_bleu4(private_plain_sequences, reconstructed_sequences)

    payload = build_security_eval_payload(
        attack="sda",
        target=SecurityEvalTarget(
            stage=resolved["stage"],
            artifact_dir=resolved["artifact_dir"],
            profile=resolved["profile"],
            model_family="qwen",
            variant=resolved["variant"],
        ),
        config={
            "phase": "gate4_minimal",
            **SDABaselineConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                knowledge_setting=knowledge_setting,
                candidate_pool_size=candidate_pool_size,
                topk=topk,
            ).to_dict(),
            "dataset_name": "phase4_local_frequency_corpora",
            "signature_type": "sorted_unigram_plus_bigram",
        },
        metrics={
            **metrics,
            "bleu4": bleu4,
            "reference_sequence_count": len(reference_sequences),
            "private_sequence_count": len(private_plain_sequences),
        },
        summary={
            "status": "completed_minimal_baseline",
            "primary_metric_name": "bleu4",
            "primary_metric_value": bleu4,
            "risk_level": classify_risk_level(metrics["token_top10_recovery_rate"]),
            "notes": "Gate-4 minimal SDA using unigram + sorted bigram signatures on local reproducible corpora.",
        },
        artifacts={
            "resolved_target": resolved,
        },
    )
    return payload


def build_sda_comparison_payload(
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
            "bleu4": metrics.get("bleu4"),
            "risk_level": summary.get("risk_level"),
            "status": summary.get("status"),
            "target_name": payload.get("artifacts", {}).get("resolved_target", {}).get("name"),
        }
        rows.append(row)
        if row["target_name"] == baseline_target_name:
            baseline_by_setting[row["knowledge_setting"]] = row["bleu4"]

    for row in rows:
        base = baseline_by_setting.get(row["knowledge_setting"])
        value = row["bleu4"]
        row["vs_stage_a_bleu4_delta"] = None if base is None or value is None else float(value - base)

    rows.sort(key=lambda item: (item["knowledge_setting"], -(item["bleu4"] or -1.0), item["stage"] or ""))
    return {
        "format": "qwen_security_sda_comparison_v1",
        "baseline_target_name": baseline_target_name,
        "row_count": len(rows),
        "rows": rows,
    }


__all__ = [
    "SDABaselineConfig",
    "build_sda_template",
    "run_sda_baseline",
    "build_sda_comparison_payload",
    "default_frequency_gate4_targets",
]
