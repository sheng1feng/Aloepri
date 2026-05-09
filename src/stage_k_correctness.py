from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.defaults import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
)
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.model_loader import set_global_seed, tokenize_prompt
from src.stage_h_pretrained import load_stage_h_pretrained
from src.stage_i_vllm import load_stage_i_hf_bundle
from src.transforms import map_input_ids, restore_logits


def resolve_stage_k_profile_paths(release_dir: str | Path, profile: str) -> dict[str, str]:
    release_dir = Path(release_dir)
    catalog = json.loads((release_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    if profile not in profile_map:
        raise ValueError(f"Unknown Stage-K profile: {profile}")
    selected = profile_map[profile]
    server_dir = release_dir / selected["server_dir"]
    profile_dir = server_dir.parent
    manifest_path = profile_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    buffered_root = manifest.get("buffered_source_of_truth") or manifest.get("resolved_source_dir") or manifest.get("standard_visible_source")
    buffered_server_dir = ""
    if isinstance(buffered_root, str) and buffered_root:
        buffered_path = Path(buffered_root)
        buffered_server_dir = str(buffered_path if buffered_path.name == "server" else buffered_path / "server")
    return {
        "server_dir": str(server_dir),
        "client_secret": str(release_dir / selected["client_secret"]),
        "correctness_evidence_file": str(selected.get("correctness_evidence_file", "")),
        "buffered_server_dir": buffered_server_dir,
    }


def summarize_prompt_results(items: list[dict[str, Any]]) -> dict[str, float | bool]:
    count = max(len(items), 1)
    return {
        "prompt_count": len(items),
        "avg_restored_full_logits_max_abs_error": sum(float(item["full_logits_max_abs_error"]) for item in items) / count,
        "avg_restored_full_logits_mean_abs_error": sum(float(item["full_logits_mean_abs_error"]) for item in items) / count,
        "avg_restored_last_token_max_abs_error": sum(float(item["last_token_logits_max_abs_error"]) for item in items) / count,
        "avg_restored_last_token_mean_abs_error": sum(float(item["last_token_logits_mean_abs_error"]) for item in items) / count,
        "greedy_first_token_match_rate": sum(1.0 for item in items if item["greedy_first_token_match"]) / count,
        "generated_ids_exact_match_rate": sum(1.0 for item in items if item["generated_ids_exact_match"]) / count,
        "generated_text_exact_match_rate": sum(1.0 for item in items if item["generated_text_exact_match"]) / count,
        "baseline_has_nan_or_inf": any(bool(item["baseline_has_nan_or_inf"]) for item in items),
        "stage_k_has_nan_or_inf": any(bool(item["stage_k_has_nan_or_inf"]) for item in items),
    }


@torch.inference_mode()
def greedy_generate_stage_k(
    model: Any,
    input_ids: torch.Tensor,
    perm_vocab: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab)
        logits_perm = model(input_ids=mapped_ids, attention_mask=torch.ones_like(mapped_ids)).logits.detach().to(torch.float32)
        restored_logits = restore_logits(logits_perm[:, -1, :], perm_vocab)
        next_token = torch.argmax(restored_logits, dim=-1, keepdim=True)
        current_plain_ids = torch.cat([current_plain_ids, next_token.to(current_plain_ids.device)], dim=1)
    return current_plain_ids[:, input_ids.shape[1] :]


def build_stage_k_correctness_summary(
    *,
    release_dir: str | Path,
    output_dir: str | Path,
    profile_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    blocking = [name for name, payload in profile_results.items() if payload.get("status") != "pass"]
    return {
        "stage": "K",
        "phase": "release_surface_correctness",
        "release_dir": str(release_dir),
        "output_dir": str(output_dir),
        "profiles": list(profile_results.keys()),
        "profile_evidence_files": {
            name: str(output_dir / f"{name}.json")
            for name in profile_results
        },
        "profile_summaries": {
            name: payload.get("summary", payload)
            for name, payload in profile_results.items()
        },
        "completion_status": "complete" if not blocking else "not_complete",
        "blocking_profiles": blocking,
    }


def run_stage_k_profile_correctness(
    *,
    release_dir: str | Path,
    profile: str,
    buffered_source_dir: str | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    set_global_seed(seed)
    profile_paths = resolve_stage_k_profile_paths(release_dir, profile)
    buffered_server_dir = buffered_source_dir or profile_paths["buffered_server_dir"]
    if not buffered_server_dir:
        raise ValueError(f"buffered source of truth is required for Stage-K correctness: {profile}")

    buffered_bundle = load_stage_h_pretrained(buffered_server_dir)
    tokenizer = buffered_bundle["tokenizer"]
    buffered_model = buffered_bundle["stage_model"].eval()
    perm_vocab = torch.as_tensor(buffered_bundle["perm_vocab"], dtype=torch.long)

    release_bundle = load_stage_i_hf_bundle(
        profile_paths["server_dir"],
        client_secret_path=profile_paths["client_secret"],
        device=device,
        dtype=dtype,
    )
    exported_tokenizer = release_bundle["tokenizer"]
    stage_model = release_bundle["model"]
    release_perm_vocab = release_bundle["perm_vocab"]

    if release_perm_vocab is None:
        raise ValueError(f"client secret is required for Stage-K correctness: {profile}")
    release_perm_vocab = torch.as_tensor(release_perm_vocab, dtype=torch.long)
    if not torch.equal(perm_vocab.cpu(), release_perm_vocab.cpu()):
        raise ValueError(f"release client secret permutation does not match buffered source of truth: {profile}")

    prompt_results: list[dict[str, Any]] = []
    for index, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        encoded = tokenize_prompt(tokenizer, prompt, device=device)
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        buffered_logits_perm = buffered_model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
        ).logits.detach().cpu().to(torch.float32)
        stage_logits_perm = stage_model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
        ).logits.detach().cpu().to(torch.float32)
        buffered_restored = restore_logits(buffered_logits_perm, perm_vocab)
        restored_logits = restore_logits(stage_logits_perm, perm_vocab)

        buffered_generated_ids = greedy_generate_stage_k(
            buffered_model,
            encoded["input_ids"],
            perm_vocab=perm_vocab,
            max_new_tokens=max_new_tokens,
        )[0].cpu()
        stage_generated_ids = greedy_generate_stage_k(
            stage_model,
            encoded["input_ids"],
            perm_vocab=perm_vocab,
            max_new_tokens=max_new_tokens,
        )[0].cpu()

        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "mapped_input_ids": mapped_ids[0].detach().cpu().tolist(),
                "full_logits_max_abs_error": max_abs_error(buffered_restored, restored_logits),
                "full_logits_mean_abs_error": mean_abs_error(buffered_restored, restored_logits),
                "last_token_logits_max_abs_error": max_abs_error(buffered_restored[0, -1], restored_logits[0, -1]),
                "last_token_logits_mean_abs_error": mean_abs_error(buffered_restored[0, -1], restored_logits[0, -1]),
                "greedy_first_token_match": int(torch.argmax(buffered_restored[0, -1]).item()) == int(torch.argmax(restored_logits[0, -1]).item()),
                "generated_ids_exact_match": buffered_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(buffered_generated_ids, skip_special_tokens=True)
                == exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "buffered_generated_ids": buffered_generated_ids.tolist(),
                "stage_k_generated_ids": stage_generated_ids.tolist(),
                "buffered_generated_text": tokenizer.decode(buffered_generated_ids, skip_special_tokens=True),
                "stage_k_generated_text": exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_has_nan_or_inf": not bool(torch.isfinite(buffered_logits_perm).all().item()),
                "stage_k_has_nan_or_inf": not bool(torch.isfinite(stage_logits_perm).all().item()),
            }
        )

    summary = summarize_prompt_results(prompt_results)
    status = (
        "pass"
        if float(summary["generated_ids_exact_match_rate"]) > 0.0
        and float(summary["generated_text_exact_match_rate"]) > 0.0
        else "fail"
    )
    return {
        "stage": "K",
        "phase": "release_surface_correctness",
        "release_dir": str(release_dir),
        "profile": profile,
        "buffered_server_dir": buffered_server_dir,
        "server_dir": profile_paths["server_dir"],
        "client_secret": profile_paths["client_secret"],
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "status": status,
        **summary,
        "summary": {"status": status, **summary},
        "prompts": prompt_results,
    }


def run_stage_k_release_correctness(
    *,
    release_dir: str | Path = "artifacts/stage_k_release",
    output_dir: str | Path = f"{DEFAULT_OUTPUT_DIR}/stage_k_release/correctness",
    profiles: tuple[str, ...] = ("default", "reference"),
    buffered_source_dir: str | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_results: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        payload = run_stage_k_profile_correctness(
            release_dir=release_dir,
            profile=profile,
            buffered_source_dir=buffered_source_dir,
            dtype=dtype,
            device=device,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        profile_results[profile] = payload
        write_json(output_dir / f"{profile}.json", payload)
    summary = build_stage_k_correctness_summary(
        release_dir=release_dir,
        output_dir=output_dir,
        profile_results=profile_results,
    )
    write_json(output_dir.parent / "correctness_summary.json", summary)
    return summary
