# Qwen Mainline Rerun Checklist Design

## 1. Context

The repository already has a consolidated Qwen mainline root:

- `docs/论文一致最终部署主线.md`

That root now correctly states:

- the active Qwen line is the paper-consistent deployment line;
- the only active release surface is `artifacts/stage_k_release`;
- the active profiles are `default` and `reference`;
- current correctness evidence is still inherited from `Stage J`;
- `VMA / IMA / ISA` still need to be rerun on the final release surface.

The user does **not** want a new side document for this next step. They want the remaining rerun work to be summarized directly inside the canonical Qwen mainline document.

The user also narrowed scope again:

- this work is only for the Qwen line;
- this step should organize the remaining reruns into an executable checklist;
- this step should **not** actually run the reruns yet.

## 2. Goals

This change must make `docs/论文一致最终部署主线.md` answer one additional question clearly:

> What exactly still needs to be rerun on the active Qwen release line, with which targets, using which existing entry points, and what evidence would count as completion?

Concretely, the update must:

1. keep all rerun planning inside the existing canonical Qwen root;
2. define the active rerun scope only in terms of `artifacts/stage_k_release`;
3. define the active security targets only in terms of `stage_k_default` and `stage_k_reference`;
4. distinguish current inherited `Stage J` evidence from future `Stage K` release-surface evidence;
5. turn “still missing reruns” into a practical checklist rather than a vague statement.

## 3. Non-Goals

This change does not:

- create a new Qwen planning document;
- rerun correctness or any attack evaluations;
- introduce a new correctness runner implementation;
- rewrite the Qwen stage reports;
- change release semantics, profile names, or target registry behavior;
- broaden scope to Llama.

## 4. Existing Repository Facts To Preserve

The checklist must be written against current repository reality.

### 4.1 Active Qwen release surface

The only active Qwen release directory is:

- `artifacts/stage_k_release`

The only active profile names are:

- `default`
- `reference`

### 4.2 Active Qwen security target names

The security target registry already exposes:

- `stage_k_default`
- `stage_k_reference`

These resolve through `artifacts/stage_k_release/catalog.json` into the active release profiles.

The checklist must not switch back to historical target names such as:

- `stage_k_tiny_a`
- `stage_j_stable_reference`
- `stage_j_tiny_a`
- redesign or bridge variants

### 4.3 Existing executable entry points

The repository already has concrete entry points for:

- release export: `scripts/export_stage_k_release.py`
- release inference by profile: `scripts/infer_stage_k_release.py`
- VMA: `scripts/security_qwen/run_vma.py`
- IMA: `scripts/security_qwen/run_ima.py`
- ISA: `scripts/security_qwen/run_isa.py`

It also has comparison/export helpers for attack summaries:

- `scripts/security_qwen/export_vma_comparison.py`
- `scripts/security_qwen/export_ima_comparison.py`
- `scripts/security_qwen/export_isa_comparison.py`

### 4.4 Current correctness evidence limitation

The active `Stage K` release catalog currently points to inherited correctness evidence:

- `outputs/stage_j/paper_consistent/correctness_regression.json`

There is currently **no separate dedicated Stage-K correctness runner or Stage-K-native correctness evidence file**.

The checklist must say this explicitly instead of implying the repository already has a finished Stage-K correctness execution path.

## 5. Proposed Documentation Change

Only one active document will be modified:

- `docs/论文一致最终部署主线.md`

No new documentation file will be added.

The root will gain a new section:

- `剩余复跑可执行清单`

This section should sit after the current evidence-entry section and before the higher-level “paper gap” and “next steps” narrative, so the document flows like this:

1. active target
2. active evidence
3. remaining executable reruns
4. current gap versus the paper
5. next-step ordering

## 6. Checklist Structure

The new checklist section must be organized into three parts.

### 6.1 Scope Lock

The section must begin by fixing the active rerun scope:

- rerun object: `artifacts/stage_k_release`
- profiles: `default`, `reference`
- security targets: `stage_k_default`, `stage_k_reference`

It must explicitly say that the following are **not** active rerun objects:

- buffered redesign
- standard-visible bridge
- historical Stage-J deployment targets

### 6.2 Correctness Rerun Checklist

This subsection must describe the remaining correctness work as an executable checklist item, but it must stay honest about the current tooling boundary.

Required content:

1. re-export or verify the active release surface with:
   - `scripts/export_stage_k_release.py`
2. smoke the actual release profiles with:
   - `scripts/infer_stage_k_release.py`
   - profile `default`
   - profile `reference`
3. state clearly that this only proves the release profiles are still runnable and user-visible;
4. state clearly that the final missing evidence is a **release-surface correctness record**, not another `Stage J` inherited pointer.

The checklist must define the desired future evidence shape explicitly, even though this change does not implement it:

- profile-level correctness evidence under `outputs/stage_k_release/correctness/`
- or an equivalent single Stage-K summary rooted under `outputs/stage_k_release/`

The exact prose should make clear:

- today’s inherited evidence is `outputs/stage_j/paper_consistent/correctness_regression.json`;
- the missing close-out artifact is a Stage-K-native correctness record for the active release surface.

### 6.3 Security Rerun Checklist

This subsection must cover only the active Qwen release targets:

- `stage_k_default`
- `stage_k_reference`

It must break the reruns into:

1. `VMA`
2. `IMA`
3. `ISA`

For each attack family, the document should name the existing runnable entry point:

- `scripts/security_qwen/run_vma.py`
- `scripts/security_qwen/run_ima.py`
- `scripts/security_qwen/run_isa.py`

It should also mention the default output conventions already implied by those scripts:

- `outputs/security_qwen/vma/stage_k_default.json`
- `outputs/security_qwen/vma/stage_k_reference.json`
- `outputs/security_qwen/ima/stage_k_default.json`
- `outputs/security_qwen/ima/stage_k_reference.json`
- `outputs/security_qwen/isa/stage_k_default.hidden_state.json`
- `outputs/security_qwen/isa/stage_k_reference.hidden_state.json`

For ISA, the checklist must explicitly say that the observable type needs to be recorded. Since the script supports both:

- `hidden_state`
- `attention_score`

the checklist must not pretend “ISA rerun” is complete unless the chosen observable is named in the evidence.

The section may mention comparison helpers as optional roll-up utilities, but per-target reruns must remain the primary active checklist items.

## 7. Completion Standard

The mainline document must define a stricter completion rule for this rerun phase.

The Qwen line may only claim that the final release surface has been rerun at the active deployment level when all of the following are true:

1. `artifacts/stage_k_release` has been regenerated or explicitly verified as the active release surface;
2. both `default` and `reference` have been smoke-checked on the release surface;
3. Stage-K-targeted `VMA` results exist for:
   - `stage_k_default`
   - `stage_k_reference`
4. Stage-K-targeted `IMA` results exist for:
   - `stage_k_default`
   - `stage_k_reference`
5. Stage-K-targeted `ISA` results exist for the explicitly named observable(s);
6. the canonical Qwen root can point to these release-surface artifacts directly, rather than only to inherited `Stage J` evidence or historical security reports.

## 8. Wording Constraints

The document update must avoid three kinds of misleading phrasing.

### 8.1 No bridge regression framing

The new checklist must not revive bridge or redesign as active rerun objects.

### 8.2 No false Stage-K correctness claim

The new checklist must not read as if a dedicated `Stage K correctness` executor already exists when it does not.

### 8.3 No Stage-J-as-final phrasing

The new checklist must not let inherited `Stage J` correctness evidence stand in for the final release-surface close-out.

## 9. Test Strategy

Implementation should extend the existing Qwen docs test coverage so the canonical root is guarded against regression.

At minimum, tests should assert that `docs/论文一致最终部署主线.md` now includes:

- a remaining-rerun checklist section;
- `artifacts/stage_k_release`;
- `stage_k_default` and `stage_k_reference`;
- `scripts/export_stage_k_release.py`;
- `scripts/infer_stage_k_release.py`;
- `scripts/security_qwen/run_vma.py`;
- `scripts/security_qwen/run_ima.py`;
- `scripts/security_qwen/run_isa.py`;
- an explicit statement that current correctness evidence is inherited from `Stage J`;
- an explicit statement that Stage-K-native correctness evidence is still missing.

Tests should also guard that the checklist does **not** present historical bridge/redesign targets as the active rerun scope.

## 10. Result

After implementation:

- the canonical Qwen root remains the only active Qwen entry point;
- the repository does not gain another planning or status document;
- the remaining Qwen reruns become actionable instead of abstract;
- the current gap between inherited `Stage J` evidence and missing `Stage K` release-surface evidence is documented precisely.
