# Stage-K Release Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage-K-native correctness evidence for the active Qwen release surface, switch the release catalog to those new evidence files, and update the canonical docs to reflect that Stage-K correctness is complete.

**Architecture:** Add a focused `src/stage_k_correctness.py` module that reuses the existing baseline-vs-obfuscated HF regression semantics against `artifacts/stage_k_release/profiles/<profile>`, wrap it with a small CLI runner that writes profile-level and release-level evidence, then repoint `src/stage_k_release.py` and the active Qwen docs to those Stage-K outputs. Keep the metric names compatible with the existing Stage-J correctness payload so old and new evidence remain directly comparable.

**Tech Stack:** Python, `pytest`, Transformers-based regression utilities, existing release catalog machinery in `src/stage_k_release.py`

---

## File Map

- Create: `src/stage_k_correctness.py`
  - Owns Stage-K profile correctness execution, release-level orchestration, and summary construction.
- Create: `scripts/run_stage_k_release_correctness.py`
  - Thin CLI entrypoint for generating Stage-K correctness evidence files.
- Create: `tests/test_stage_k_correctness.py`
  - Owns unit coverage for Stage-K correctness path resolution, payload construction, and release summary generation.
- Modify: `src/stage_k_release.py`
  - Switch active profile correctness pointers from Stage-J evidence to Stage-K evidence and normalize correctness summaries.
- Modify: `tests/test_stage_k_release.py`
  - Update release-catalog expectations to the new Stage-K correctness paths and summary behavior.
- Modify: `docs/论文一致最终部署主线.md`
  - Change the canonical Qwen root from “Stage-K correctness missing” to “Stage-K correctness complete” and point to the new evidence files.
- Modify: `docs/阶段K_Qwen交付包装报告.md`
  - Change the Stage-K report to point at Stage-K-native correctness evidence instead of inherited Stage-J correctness.
- Modify: `tests/test_qwen_paper_consistent_docs.py`
  - Update canonical-doc assertions to the new correctness paths and wording.

### Task 1: Add Failing Tests For Stage-K-Native Correctness

**Files:**
- Create: `tests/test_stage_k_correctness.py`
- Modify: `tests/test_stage_k_release.py:30-95`
- Modify: `tests/test_qwen_paper_consistent_docs.py:112-201`

- [ ] **Step 1: Write the failing correctness unit tests**

Create `tests/test_stage_k_correctness.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

from src.stage_k_correctness import (
    build_stage_k_correctness_summary,
    resolve_stage_k_profile_paths,
    summarize_prompt_results,
)


def test_resolve_stage_k_profile_paths_reads_catalog_profile_dirs(tmp_path: Path) -> None:
    release_dir = tmp_path / "stage_k_release"
    (release_dir / "profiles" / "default" / "server").mkdir(parents=True)
    (release_dir / "profiles" / "default" / "client").mkdir(parents=True)
    (release_dir / "profiles" / "default" / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")
    (release_dir / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "default",
                        "server_dir": "profiles/default/server",
                        "client_secret": "profiles/default/client/client_secret.pt",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = resolve_stage_k_profile_paths(release_dir, "default")
    assert payload["server_dir"] == str(release_dir / "profiles" / "default" / "server")
    assert payload["client_secret"] == str(release_dir / "profiles" / "default" / "client" / "client_secret.pt")


def test_summarize_prompt_results_uses_stage_j_compatible_metrics() -> None:
    summary = summarize_prompt_results(
        [
            {
                "full_logits_max_abs_error": 0.5,
                "full_logits_mean_abs_error": 0.25,
                "last_token_logits_max_abs_error": 0.2,
                "last_token_logits_mean_abs_error": 0.1,
                "greedy_first_token_match": True,
                "generated_ids_exact_match": True,
                "generated_text_exact_match": False,
                "baseline_has_nan_or_inf": False,
                "stage_k_has_nan_or_inf": False,
            },
            {
                "full_logits_max_abs_error": 0.25,
                "full_logits_mean_abs_error": 0.125,
                "last_token_logits_max_abs_error": 0.1,
                "last_token_logits_mean_abs_error": 0.05,
                "greedy_first_token_match": False,
                "generated_ids_exact_match": True,
                "generated_text_exact_match": True,
                "baseline_has_nan_or_inf": False,
                "stage_k_has_nan_or_inf": False,
            },
        ]
    )

    assert summary["prompt_count"] == 2
    assert summary["generated_ids_exact_match_rate"] == 1.0
    assert summary["generated_text_exact_match_rate"] == 0.5


def test_build_stage_k_correctness_summary_wraps_profile_results() -> None:
    release_dir = "artifacts/stage_k_release"
    output_dir = "outputs/stage_k_release/correctness"
    summary = build_stage_k_correctness_summary(
        release_dir=release_dir,
        output_dir=output_dir,
        profile_results={
            "default": {
                "status": "pass",
                "prompt_count": 5,
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
            },
            "reference": {
                "status": "pass",
                "prompt_count": 5,
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
            },
        },
    )

    assert summary["release_dir"] == release_dir
    assert summary["profiles"] == ["default", "reference"]
    assert summary["completion_status"] == "complete"
    assert summary["profile_evidence_files"]["default"] == f"{output_dir}/default.json"
```

- [ ] **Step 2: Update failing release/docs tests for Stage-K evidence paths**

Patch `tests/test_stage_k_release.py` and `tests/test_qwen_paper_consistent_docs.py` so they expect Stage-K-native correctness:

```python
def test_stage_k_profiles_point_to_stage_k_correctness_evidence() -> None:
    profiles = default_stage_k_profiles()
    assert [item.correctness_evidence_file for item in profiles] == [
        "outputs/stage_k_release/correctness/default.json",
        "outputs/stage_k_release/correctness/reference.json",
    ]
```

```python
def test_stage_k_docs_use_stage_k_correctness_release_surface() -> None:
    stage_k_text = Path("docs/阶段K_Qwen交付包装报告.md").read_text(encoding="utf-8")
    main_text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    assert "outputs/stage_k_release/correctness/default.json" in stage_k_text
    assert "outputs/stage_k_release/correctness/reference.json" in stage_k_text
    assert "outputs/stage_k_release/correctness_summary.json" in stage_k_text
    assert "outputs/stage_k_release/correctness/default.json" in main_text
    assert "outputs/stage_k_release/correctness/reference.json" in main_text
```

Also update the rerun-checklist assertions to require Stage-K correctness paths and remove the “still missing” wording check.

- [ ] **Step 3: Run the new/updated tests to verify RED**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q \
  tests/test_stage_k_correctness.py \
  tests/test_stage_k_release.py \
  tests/test_qwen_paper_consistent_docs.py
```

Expected:

- FAIL because `src/stage_k_correctness.py` does not exist yet
- FAIL because release/docs still point to Stage-J correctness paths

- [ ] **Step 4: Commit the red test state after implementation passes, not before**

No commit in red state.

### Task 2: Implement Stage-K Correctness Module And CLI

**Files:**
- Create: `src/stage_k_correctness.py`
- Create: `scripts/run_stage_k_release_correctness.py`
- Test: `tests/test_stage_k_correctness.py`

- [ ] **Step 1: Write the minimal Stage-K correctness implementation**

Create `src/stage_k_correctness.py` with this structure:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.model_loader import load_model_and_tokenizer, set_global_seed, tokenize_prompt
from src.stage_i_vllm import load_stage_i_hf_bundle
from src.transforms import map_input_ids, restore_logits


def resolve_stage_k_profile_paths(release_dir: str | Path, profile: str) -> dict[str, str]:
    release_dir = Path(release_dir)
    catalog = json.loads((release_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    selected = profile_map[profile]
    return {
        "server_dir": str(release_dir / selected["server_dir"]),
        "client_secret": str(release_dir / selected["client_secret"]),
    }


def summarize_prompt_results(items: list[dict[str, Any]]) -> dict[str, float | bool]:
    count = max(len(items), 1)
    return {
        "prompt_count": len(items),
        "avg_full_logits_max_abs_error": sum(float(item["full_logits_max_abs_error"]) for item in items) / count,
        "avg_full_logits_mean_abs_error": sum(float(item["full_logits_mean_abs_error"]) for item in items) / count,
        "avg_last_token_logits_max_abs_error": sum(float(item["last_token_logits_max_abs_error"]) for item in items) / count,
        "avg_last_token_logits_mean_abs_error": sum(float(item["last_token_logits_mean_abs_error"]) for item in items) / count,
        "greedy_first_token_match_rate": sum(1.0 for item in items if item["greedy_first_token_match"]) / count,
        "generated_ids_exact_match_rate": sum(1.0 for item in items if item["generated_ids_exact_match"]) / count,
        "generated_text_exact_match_rate": sum(1.0 for item in items if item["generated_text_exact_match"]) / count,
        "baseline_has_nan_or_inf": any(bool(item["baseline_has_nan_or_inf"]) for item in items),
        "stage_k_has_nan_or_inf": any(bool(item["stage_k_has_nan_or_inf"]) for item in items),
    }
```

- [ ] **Step 2: Add profile execution and release summary writing**

Append the execution functions to `src/stage_k_correctness.py`:

```python
@torch.inference_mode()
def greedy_generate_plain(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(input_ids=current_ids, attention_mask=torch.ones_like(current_ids)).logits.detach()
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token.to(current_ids.device)], dim=1)
    return current_ids[:, input_ids.shape[1] :]


@torch.inference_mode()
def greedy_generate_stage_k(model, input_ids: torch.Tensor, perm_vocab: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab)
        logits_perm = model(input_ids=mapped_ids, attention_mask=torch.ones_like(mapped_ids)).logits.detach().cpu().to(torch.float32)
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
        "profiles": list(profile_results.keys()),
        "profile_evidence_files": {
            name: str(output_dir / f"{name}.json")
            for name in profile_results
        },
        "profile_summaries": {
            name: payload["summary"]
            for name, payload in profile_results.items()
        },
        "completion_status": "complete" if not blocking else "not_complete",
        "blocking_profiles": blocking,
    }
```

```python
def run_stage_k_profile_correctness(
    *,
    release_dir: str | Path,
    profile: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    dtype: str = "float32",
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    set_global_seed(seed)
    profile_paths = resolve_stage_k_profile_paths(release_dir, profile)
    tokenizer, baseline_model = load_model_and_tokenizer(baseline_model_dir, device=device, dtype=dtype)
    bundle = load_stage_i_hf_bundle(
        profile_paths["server_dir"],
        client_secret_path=profile_paths["client_secret"],
        device=device,
        dtype=dtype,
    )
    exported_tokenizer = bundle["tokenizer"]
    stage_model = bundle["model"]
    perm_vocab = bundle["perm_vocab"]
    if perm_vocab is None:
        raise ValueError(f"client secret is required for Stage-K correctness: {profile}")

    prompt_results: list[dict[str, Any]] = []
    for index, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        encoded = tokenize_prompt(tokenizer, prompt, device=device)
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        stage_logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded["attention_mask"]).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(stage_logits_perm, perm_vocab)
        baseline_generated_ids = greedy_generate_plain(baseline_model, encoded["input_ids"], max_new_tokens=max_new_tokens)[0].cpu()
        stage_generated_ids = greedy_generate_stage_k(stage_model, encoded["input_ids"], perm_vocab=perm_vocab, max_new_tokens=max_new_tokens)[0].cpu()
        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "full_logits_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "full_logits_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "last_token_logits_max_abs_error": max_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "last_token_logits_mean_abs_error": mean_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "greedy_first_token_match": int(torch.argmax(baseline_logits[0, -1]).item()) == int(torch.argmax(restored_logits[0, -1]).item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_generated_ids": baseline_generated_ids.tolist(),
                "stage_k_generated_ids": stage_generated_ids.tolist(),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "stage_k_generated_text": exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_has_nan_or_inf": not bool(torch.isfinite(baseline_logits).all().item()),
                "stage_k_has_nan_or_inf": not bool(torch.isfinite(stage_logits_perm).all().item()),
            }
        )

    summary = summarize_prompt_results(prompt_results)
    status = "pass" if float(summary["generated_ids_exact_match_rate"]) > 0.0 and float(summary["generated_text_exact_match_rate"]) > 0.0 else "fail"
    return {
        "stage": "K",
        "phase": "release_surface_correctness",
        "release_dir": str(release_dir),
        "profile": profile,
        "server_dir": profile_paths["server_dir"],
        "client_secret": profile_paths["client_secret"],
        "baseline_model_dir": baseline_model_dir,
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "status": status,
        **summary,
        "summary": {"status": status, **summary},
        "prompts": prompt_results,
    }
```

```python
def run_stage_k_release_correctness(
    *,
    release_dir: str | Path = "artifacts/stage_k_release",
    output_dir: str | Path = f"{DEFAULT_OUTPUT_DIR}/stage_k_release/correctness",
    profiles: tuple[str, ...] = ("default", "reference"),
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
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
            baseline_model_dir=baseline_model_dir,
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
    write_json(Path(output_dir).parent / "correctness_summary.json", summary)
    return summary
```

- [ ] **Step 3: Add the CLI runner**

Create `scripts/run_stage_k_release_correctness.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.stage_k_correctness import run_stage_k_release_correctness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-K release-surface correctness for active Qwen profiles.")
    parser.add_argument("--release-dir", default="artifacts/stage_k_release")
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_DIR}/stage_k_release/correctness")
    parser.add_argument("--profiles", nargs="*", default=["default", "reference"])
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage_k_release_correctness(
        release_dir=args.release_dir,
        output_dir=args.output_dir,
        profiles=tuple(args.profiles),
        baseline_model_dir=args.baseline_model_dir,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Saved Stage-K correctness summary to {Path(args.output_dir).parent / 'correctness_summary.json'}")
    print(f"Profiles: {summary['profiles']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the new correctness tests to verify GREEN**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_stage_k_correctness.py
```

Expected:

- PASS

- [ ] **Step 5: Commit the new correctness module**

Run:

```bash
git add src/stage_k_correctness.py scripts/run_stage_k_release_correctness.py tests/test_stage_k_correctness.py
git commit -m "feat: add stage k release correctness runner"
```

### Task 3: Repoint Release Catalog, Re-export Stage-K, And Update Docs

**Files:**
- Modify: `src/stage_k_release.py:19-144`
- Modify: `tests/test_stage_k_release.py:20-95`
- Modify: `docs/论文一致最终部署主线.md:63-203`
- Modify: `docs/阶段K_Qwen交付包装报告.md:29-42`
- Modify: `tests/test_qwen_paper_consistent_docs.py:128-201`

- [ ] **Step 1: Update Stage-K profile pointers and correctness-summary normalization**

Patch `src/stage_k_release.py`:

```python
def default_stage_k_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="default",
            source_dir="artifacts/stage_j_qwen_paper_consistent",
            description="Default paper-consistent Stage-J Qwen release profile.",
            recommended_use="Default delivery entry for the paper-consistent Qwen deployment line.",
            correctness_evidence_file="outputs/stage_k_release/correctness/default.json",
        ),
        StageKProfile(
            name="reference",
            source_dir="artifacts/stage_j_qwen_paper_consistent",
            description="Reference paper-consistent Stage-J Qwen release profile.",
            recommended_use="Audit and evidence entry for the same paper-consistent deployment line.",
            correctness_evidence_file="outputs/stage_k_release/correctness/reference.json",
        ),
    ]


def _read_correctness_summary(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _load_json(Path(path))
    if not payload:
        return {}
    nested = payload.get("summary")
    if isinstance(nested, dict):
        if "status" not in nested and "status" in payload:
            return {"status": payload["status"], **nested}
        return nested
    return payload
```

Then update `_profile_summary(...)` to use `_read_correctness_summary(profile.correctness_evidence_file)` instead of directly loading the whole payload.

- [ ] **Step 2: Update canonical docs and Stage-K report**

Adjust the docs so they point to Stage-K-native correctness and no longer say it is missing.

Required doc changes:

```md
- `Stage K` correctness evidence：
  - `outputs/stage_k_release/correctness/default.json`
  - `outputs/stage_k_release/correctness/reference.json`
  - `outputs/stage_k_release/correctness_summary.json`
```

And in the canonical Qwen root, change the remaining-work narrative so the post-correctness remaining work is:

```md
1. 在唯一 release 面上复跑 `VMA / IMA / ISA`
2. 根据复跑结果决定是否需要细化 `default` / `reference` 的独立工作点语义
```

- [ ] **Step 3: Run the real Stage-K correctness job and re-export the release**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_k_release_correctness.py \
  --release-dir artifacts/stage_k_release \
  --output-dir outputs/stage_k_release/correctness

conda run --no-capture-output -n qwen-transformers python scripts/export_stage_k_release.py \
  --export-dir artifacts/stage_k_release
```

Expected:

- `outputs/stage_k_release/correctness/default.json` exists
- `outputs/stage_k_release/correctness/reference.json` exists
- `outputs/stage_k_release/correctness_summary.json` exists
- `artifacts/stage_k_release/catalog.json` now points to the Stage-K correctness files

- [ ] **Step 4: Run the updated regression suite**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q \
  tests/test_stage_k_correctness.py \
  tests/test_stage_k_release.py \
  tests/test_qwen_paper_consistent_docs.py \
  tests/test_mainline_docs_history.py \
  tests/test_stage_hk_alignment_checklist.py \
  tests/test_security_qwen_summary.py \
  tests/test_stage_k_llama_release.py

git diff --check
```

Expected:

- all listed tests PASS
- `git diff --check` prints nothing

- [ ] **Step 5: Commit the release cutover and docs update**

Run:

```bash
git add \
  src/stage_k_release.py \
  tests/test_stage_k_release.py \
  docs/论文一致最终部署主线.md \
  docs/阶段K_Qwen交付包装报告.md \
  tests/test_qwen_paper_consistent_docs.py \
  artifacts/stage_k_release/catalog.json \
  artifacts/stage_k_release/README.md \
  outputs/stage_k_release/correctness/default.json \
  outputs/stage_k_release/correctness/reference.json \
  outputs/stage_k_release/correctness_summary.json
git commit -m "feat: add stage k release correctness evidence"
```
