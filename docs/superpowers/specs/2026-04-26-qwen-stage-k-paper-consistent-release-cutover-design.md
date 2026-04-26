# Qwen Stage K Paper-Consistent Release Cutover Design

## 1. Goal

Cut over the active Qwen `Stage K` release surface to the single paper-consistent deployment line that was established in `Stage J`.

After this round:

- `artifacts/stage_k_release` remains the only active Qwen release directory;
- its active profiles become `default` and `reference`;
- both profiles resolve to the same canonical `Stage J` source artifact: `artifacts/stage_j_qwen_paper_consistent`;
- active code, tests, and security target registration stop treating `stable_reference` / `tiny_a` as current `Stage K` semantics.

## 2. Current Problem

`Stage J` has already been recentered onto:

- `artifacts/stage_j_qwen_paper_consistent`
- `outputs/stage_j/paper_consistent/completion_summary.json`

But the active `Stage K` release implementation still packages the older redesign line:

- `src/stage_k_release.py` still emits `stable_reference` and `tiny_a`;
- `artifacts/stage_k_release/catalog.json` still declares `stage_lineage = redesigned_qwen_stage_j`;
- active security targets still point `Stage K` at the old release-profile naming and release semantics.

This leaves the repository in a split state where `Stage J` and `Stage K` no longer describe the same canonical deployment line.

## 3. Decision

Use a full active-surface cutover.

The repository will not keep a parallel new release directory and will not preserve active compatibility aliases for old `Stage K` profile names. Instead:

- keep `artifacts/stage_k_release` as the single active release directory;
- rename active profiles to `default` and `reference`;
- point both profiles to `artifacts/stage_j_qwen_paper_consistent`;
- replace active `Stage K` security targets with `stage_k_default` and `stage_k_reference`;
- update active docs and tests to match the new release semantics.

Historical `Stage J` and security result documents are not renamed in this round. They remain historical evidence, not active release definitions.

## 4. Release Shape

### 4.1 Active directory

The active directory remains:

- `artifacts/stage_k_release`

No second canonical release directory is introduced.

### 4.2 Active profiles

The active release profiles become:

- `default`
- `reference`

Semantics:

- `default`: the default delivery entry for the paper-consistent deployment line
- `reference`: the audit/evidence-facing entry for the same canonical deployment line

In this round, both profiles point to the same source artifact:

- `artifacts/stage_j_qwen_paper_consistent`

They are semantic aliases, not distinct model variants.

### 4.3 Catalog fields

The exported `catalog.json` must reflect the new semantics:

- `stage_lineage = paper_consistent_stage_j`
- `recommended_profile = default`
- `reference_profile = reference`

The old field `stable_reference_profile` is removed from the active catalog shape.

### 4.4 Release README semantics

The release README must say that the bundle collects the paper-consistent Qwen deployment line, not the redesign line.

## 5. Code Changes

### 5.1 Stage K exporter

Update `src/stage_k_release.py` so that:

- `default_stage_k_profiles()` returns only `default` and `reference`;
- both profile entries use `source_dir="artifacts/stage_j_qwen_paper_consistent"`;
- descriptions and recommended-use text reflect paper-consistent release semantics;
- the catalog uses `reference_profile` instead of `stable_reference_profile`;
- lineage and README strings refer to the paper-consistent release line.

### 5.2 Export CLI

Keep `scripts/export_stage_k_release.py` as the active export entrypoint.

It should continue writing:

- `artifacts/stage_k_release`

But the generated content must now match the paper-consistent profile and lineage semantics.

### 5.3 Inference CLI

Update `scripts/infer_stage_k_release.py` so its default profile becomes:

- `default`

The rest of the lookup behavior remains catalog-driven.

## 6. Security Target Cutover

### 6.1 Active Stage K targets

Active `Stage K` target names become:

- `stage_k_default`
- `stage_k_reference`

The following active names are removed from code-path registration:

- `stage_k_stable_reference`
- `stage_k_tiny_a`
- `stage_k_redesign_stable_reference`
- `stage_k_redesign_tiny_a`

### 6.2 Resolution behavior

Both active `Stage K` targets must resolve through `artifacts/stage_k_release/catalog.json` into:

- `artifacts/stage_k_release/profiles/default`
- `artifacts/stage_k_release/profiles/reference`

### 6.3 Security matrix

Any active matrix or summary builder that currently enumerates `stage_k_stable_reference` or `stage_k_tiny_a` must be switched to `stage_k_reference` and `stage_k_default`.

### 6.4 Stage H-K audit

`src/stage_hk_audit.py` must stop treating `Stage K` as a redesign release wrapper and instead verify that the active release points to the paper-consistent `Stage J` artifact.

## 7. Documentation Changes

Update the active Qwen-facing docs so they no longer present `Stage K` as a redesign-line release:

- `docs/阶段K_Qwen交付包装报告.md`
- `docs/论文一致最终部署主线.md`
- `README.md`

The docs must state that:

- `artifacts/stage_k_release` is the single active Qwen release surface;
- `default` is the default delivery profile;
- `reference` is the audit/evidence profile;
- both derive from `artifacts/stage_j_qwen_paper_consistent`;
- legacy `Stage K` naming is historical only.

Historical security result documents may continue using old experiment names because they record prior runs, not the new canonical release semantics.

## 8. Tests

The round must update or add tests that prove:

1. `default_stage_k_profiles()` returns `default` and `reference`
2. both active profiles point to `artifacts/stage_j_qwen_paper_consistent`
3. `resolve_security_target()` maps `stage_k_default` and `stage_k_reference` to the expected profile directories
4. the generated `artifacts/stage_k_release/catalog.json` advertises:
   - `paper_consistent_stage_j`
   - `recommended_profile = default`
   - `reference_profile = reference`
5. active docs no longer describe `Stage K` as the redesign release line

Relevant test surfaces include:

- `tests/test_stage_k_release.py`
- `tests/test_stage_k_redesign.py`
- `tests/test_qwen_paper_consistent_docs.py`
- active security summary / target resolution tests

## 9. Verification

Completion requires fresh verification of:

- focused pytest coverage for updated `Stage K` and security tests;
- successful execution of `scripts/export_stage_k_release.py`;
- regenerated `artifacts/stage_k_release/catalog.json` with the new profile names and lineage;
- `git diff --check` clean.

## 10. Out of Scope

This round does not:

- create differentiated paper-consistent `default` and `reference` model variants;
- rename historical `Stage J` experiment documents or security result tables;
- re-run the full `VMA / IMA / ISA / TFMA / SDA` suite on the new `Stage K` release surface.

Those can follow after the release cutover is complete.
