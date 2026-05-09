# Stage-K Release Correctness Design

## 1. Context

The active Qwen deployment line is now:

- `artifacts/stage_j_qwen_paper_consistent`
- packaged as the only active release surface: `artifacts/stage_k_release`
- with two active profiles: `default` and `reference`

The repository already completed two documentation cleanups:

1. the canonical Qwen root is `docs/论文一致最终部署主线.md`;
2. that root now includes an executable checklist for the remaining `Stage K` reruns.

At the moment, the release surface still inherits correctness evidence from `Stage J`:

- `outputs/stage_j/paper_consistent/correctness_regression.json`

That is no longer sufficient. The active `Stage K` release surface needs its **own** correctness evidence so the catalog, release README, and mainline docs can stop pointing to inherited `Stage J` correctness.

The user chose these boundaries:

- this step is only about `Stage K` release-surface correctness;
- run correctness for **both** `default` and `reference`;
- keep the same correctness semantics as the current `Stage J` evidence;
- produce profile-level files **and** a release-level summary;
- switch `artifacts/stage_k_release/catalog.json` so profile correctness pointers reference the new `Stage K` evidence.

## 2. Goals

This work must:

1. create runnable correctness evaluation for `artifacts/stage_k_release` profiles;
2. produce `Stage K`-native correctness evidence for `default` and `reference`;
3. produce one release-level summary aggregating those profile results;
4. update the `Stage K` catalog so correctness pointers no longer reference `Stage J`;
5. update Qwen documentation and tests so the active line reflects the new evidence location;
6. preserve direct comparability with current `Stage J` correctness semantics.

## 3. Non-Goals

This work does not:

- run `VMA / IMA / ISA`;
- redesign the Qwen release profiles;
- create a new deployment-line root;
- change the active `default` / `reference` profile names;
- introduce a brand-new correctness metric family;
- change Llama release semantics.

## 4. Existing Facts To Preserve

### 4.1 Release shape

The active release catalog format is defined by `src/stage_k_release.py`.

Each profile currently carries:

- `name`
- `source_dir`
- `description`
- `recommended_use`
- `correctness_evidence_file`

The catalog summary also reads the pointed correctness file and embeds it as:

- `correctness_summary`

### 4.2 Current correctness semantics

The current Qwen correctness line already uses these summary fields:

- `prompt_count`
- `generated_ids_exact_match_rate`
- `generated_text_exact_match_rate`

In practice, the underlying regression payloads also carry:

- full/last-token logits max/mean absolute error
- greedy first-token match
- per-prompt generated token/text comparisons

The new `Stage K` evidence should preserve this shape rather than inventing a new smoke-only payload.

### 4.3 Stable prompt source

The repository already has a stable prompt set through:

- `src.defaults.DEFAULT_PROMPTS`

The `Stage K` correctness run should reuse the same prompt source so the new evidence remains directly comparable to the current `Stage J` correctness output.

## 5. Evidence Shape

This work will create three new active evidence files:

- `outputs/stage_k_release/correctness/default.json`
- `outputs/stage_k_release/correctness/reference.json`
- `outputs/stage_k_release/correctness_summary.json`

### 5.1 Profile-level files

`default.json` and `reference.json` must each record correctness for the corresponding release profile under:

- `artifacts/stage_k_release/profiles/default`
- `artifacts/stage_k_release/profiles/reference`

Each payload should include:

- stage identifier (`K`)
- release directory
- profile name
- server directory
- client secret path
- baseline model directory
- dtype / device / seed / max_new_tokens
- a `summary` object with the Stage-J-compatible metrics
- a `prompts` array with per-prompt detailed results

### 5.2 Release-level summary

`outputs/stage_k_release/correctness_summary.json` must summarize both profile results in one active entry.

It should include:

- release directory
- profiles evaluated
- per-profile evidence file paths
- per-profile `summary`
- a top-level completion verdict for the release-surface correctness phase

It does not need to invent cross-profile aggregate math beyond surfacing the two profile summaries. A small wrapper summary is sufficient.

## 6. Execution Model

### 6.1 Scope of evaluation

The correctness run must evaluate:

- `default`
- `reference`

No historical `Stage J`, redesign, or bridge target should be treated as the active evaluation object in this step.

### 6.2 Reuse existing regression semantics

The new execution path should reuse the same baseline-vs-obfuscated comparison style already used by current HF regression scripts:

- same prompt source
- same generation-comparison semantics
- same summary field names for the core correctness rates

The only change in execution object is:

- instead of pointing directly at a Stage-I or Stage-J exported bundle,
- load the server/client material from `artifacts/stage_k_release/profiles/<profile>/...`

### 6.3 Minimal new module surface

Add a focused module:

- `src/stage_k_correctness.py`

This module should own only:

- profile-level correctness execution
- release-level orchestration across profiles
- release-level summary construction

The CLI wrapper should remain thin:

- `scripts/run_stage_k_release_correctness.py`

## 7. Catalog And Release Update

After correctness files exist, `src/stage_k_release.py` must point active profile correctness to the new `Stage K` evidence:

- `default` -> `outputs/stage_k_release/correctness/default.json`
- `reference` -> `outputs/stage_k_release/correctness/reference.json`

Then:

- re-export `artifacts/stage_k_release`
- regenerate `catalog.json`
- regenerate `README.md`

After this change, the active `Stage K` release should no longer advertise:

- `outputs/stage_j/paper_consistent/correctness_regression.json`

as its primary correctness evidence.

## 8. Documentation Update

The canonical Qwen root and the `Stage K` report must be updated.

### 8.1 `docs/论文一致最终部署主线.md`

The rerun checklist and paper-gap sections must change from:

- “`Stage K` correctness is still missing”

to:

- `Stage K` correctness evidence exists at the new release paths;
- the remaining active work after this step is `VMA / IMA / ISA`.

### 8.2 `docs/阶段K_Qwen交付包装报告.md`

This document must stop describing `Stage K` correctness as inherited from `Stage J` only.

It should instead point at:

- `outputs/stage_k_release/correctness/default.json`
- `outputs/stage_k_release/correctness/reference.json`
- `outputs/stage_k_release/correctness_summary.json`

## 9. Test Strategy

Implementation must follow TDD.

### 9.1 New tests

Create:

- `tests/test_stage_k_correctness.py`

This file should cover:

- profile payload construction
- release summary construction
- output path conventions

### 9.2 Updated tests

Update:

- `tests/test_stage_k_release.py`
- `tests/test_qwen_paper_consistent_docs.py`

The tests should assert:

- active Stage-K profile correctness pointers now reference `outputs/stage_k_release/correctness/*.json`
- the release summary is read into the catalog correctly
- the canonical Qwen doc points to Stage-K-native correctness evidence
- the canonical Qwen doc no longer describes Stage-K correctness as still missing

## 10. Verification

This work is only complete when all of the following succeed:

1. unit tests for the new Stage-K correctness module;
2. updated Stage-K release tests;
3. updated canonical-doc tests;
4. actual execution of `scripts/run_stage_k_release_correctness.py`;
5. actual re-export of `artifacts/stage_k_release`;
6. `git diff --check` clean.

## 11. Result

After this step:

- `Stage K` has its own correctness evidence;
- the release catalog points to active `Stage K` evidence instead of inherited `Stage J` evidence;
- the Qwen mainline can describe `Stage K correctness` as complete;
- the only remaining active rerun class in the canonical Qwen line is `VMA / IMA / ISA`.
