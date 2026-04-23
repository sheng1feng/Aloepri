# Qwen Stage J Export-Visible Completion Design

## 1. Goal

Close `Stage J` at the level of **export-visible evidence**, not merely at the level of a loadable bridge artifact.

The concrete goal of this round is to:

- export one new canonical `paper-consistent` `Stage J` candidate directly from `artifacts/stage_j_qwen_redesign`;
- prove that the candidate is a standard-weight deployment artifact rather than a buffered internal-state artifact;
- attach one unified evidence package covering `attention`, `FFN`, `norm`, and correctness;
- update the canonical `Stage J` narrative so the new candidate becomes the only active `Stage J` acceptance target.

This round does **not** switch `Stage K` to the new line.

## 2. Problem

The repository already contains several partial `Stage J` abilities:

- `standard_weight_proof` proves whether an exported artifact uses standard `model.* / lm_head.*` keys;
- `attention_gap`, `norm_gap`, and `component_gap` audit specific component deficits;
- `bridge_regression` compares the historical `standard-visible bridge` against the buffered redesign source.

However, these abilities are still organized around intermediate lines:

- `artifacts/stage_j_qwen_redesign` acts as the buffered redesign source of truth;
- `artifacts/stage_j_qwen_redesign_standard` acts as a historical standard-visible bridge;
- the current docs explicitly state that neither one is the final paper-consistent `Stage J` answer.

As a result, the repository can currently prove:

- that a historical bridge is loadable and standard-key-visible;
- that specific gaps still exist;
- that the bridge is not semantically equivalent to the redesign source.

It cannot yet prove:

- that one canonical `Stage J` candidate exists;
- that the candidate itself carries export-visible evidence for the paper-relevant component expressions;
- that the candidate is the new `Stage J` acceptance target.

## 3. Scope

### In scope

- Define one new canonical candidate artifact:
  - `artifacts/stage_j_qwen_paper_consistent/`
- Define one new unified evidence directory:
  - `outputs/stage_j/paper_consistent/`
- Add export logic that materializes the canonical candidate from `artifacts/stage_j_qwen_redesign`
- Add unified proof/report generation for:
  - standard weight visibility
  - attention export-visible proof
  - FFN export-visible proof
  - norm export-visible proof
  - correctness regression
- Add or update tests so `Stage J` completion is checked against the canonical candidate and unified evidence package
- Update canonical docs so `Stage J` centers on the new candidate rather than the historical bridge

### Out of scope

- Switching `Stage K` release packaging to the new candidate
- Re-running full `VMA / IMA / ISA` on the final line
- Deleting historical `redesign_standard` tooling or artifacts
- Declaring token-perfect equivalence if the evidence only supports export-visible completion

## 4. Chosen Approach

Three approaches were considered:

1. Continue improving `artifacts/stage_j_qwen_redesign_standard`
2. Export a new canonical final candidate directly from `artifacts/stage_j_qwen_redesign`
3. Only improve audits without exporting a new candidate

This design chooses **Approach 2**.

Reasons:

- the current bridge line is semantically framed as a bridge, not as a final answer;
- the paper-consistent target should be built directly as a standard deployment artifact, not described as "bridge to redesign";
- export-visible completion requires a single candidate and a single evidence package, which fits a direct-export design better than bridge patching.

## 5. Artifact Model

### 5.1 Source artifact

`artifacts/stage_j_qwen_redesign/`

Role:

- remains the source-of-truth artifact for the redesigned `Stage J` expression inventory;
- remains the comparison baseline for correctness and export-visible proof generation;
- is no longer treated as the deployable final answer.

### 5.2 Canonical candidate artifact

`artifacts/stage_j_qwen_paper_consistent/`

Role:

- becomes the only active `Stage J` candidate under the canonical paper-consistent line;
- must present as a standard deployment artifact with standard weight layout;
- must carry enough manifest/proof metadata to support export-visible evidence.

Expected shape:

- `artifacts/stage_j_qwen_paper_consistent/server/`
- `artifacts/stage_j_qwen_paper_consistent/client/client_secret.pt`
- `artifacts/stage_j_qwen_paper_consistent/manifest.json`

### 5.3 Historical bridge artifact

`artifacts/stage_j_qwen_redesign_standard/`

Role after this round:

- historical intermediate artifact only;
- retained for comparison and backward reference;
- no longer treated as the canonical `Stage J` acceptance target.

## 6. Unified Evidence Package

All new `Stage J` completion evidence must be generated under:

- `outputs/stage_j/paper_consistent/`

The package must contain at least these files:

- `standard_weight_proof.json`
- `attention_export_visible_proof.json`
- `ffn_export_visible_proof.json`
- `norm_export_visible_proof.json`
- `correctness_regression.json`
- `completion_summary.json`

### 6.1 Standard weight proof

Question answered:

> Does the canonical candidate export as a standard `model.* / lm_head.*` checkpoint rather than a buffered stage-style checkpoint?

Minimum pass conditions:

- `is_standard_weight_export = true`
- `layout = standard_weight_visible`
- key standard weight groups required by the model family are present

Fail meaning:

- the candidate is not yet a paper-consistent deployment artifact.

### 6.2 Attention export-visible proof

Question answered:

> Have the paper-relevant attention expressions been materialized into the exported standard-weight candidate, rather than remaining only inside buffered redesign internals?

Minimum pass conditions:

- exported candidate is backed by explicit attention-expression metadata and proof fields, not just generic standard `qkv/o` weights;
- proof covers the presence or carried-forward status of paper-relevant attention-side expressions:
  - attention profile
  - head/group/block diversity or permutation semantics
  - `q/k`-side structured transforms or their exported-equivalent representation
- proof produces an explicit pass/fail judgment instead of a narrative-only note.

Fail meaning:

- the candidate is still only "attention shell standardized" rather than attention export-visible complete.

### 6.3 FFN export-visible proof

Question answered:

> Have the paper-relevant FFN component transforms been materialized into the exported standard-weight candidate?

Minimum pass conditions:

- proof explicitly covers `gate`, `up`, and `down` component semantics;
- proof is component-level rather than manifest-only;
- proof produces an explicit pass/fail judgment.

Fail meaning:

- FFN remains an unclosed export-visible gap.

### 6.4 Norm export-visible proof

Question answered:

> Does the candidate provide a concrete exported-standard-artifact answer for the paper-relevant norm correction semantics, rather than falling back to a placeholder or bridge heuristic?

Minimum pass conditions:

- proof states how norm semantics are carried by the candidate artifact;
- proof distinguishes a real exported-standard answer from a placeholder bridge approximation;
- proof produces an explicit pass/fail judgment.

Fail meaning:

- norm remains a structural blocker for `Stage J` completion.

### 6.5 Correctness regression

Question answered:

> Is the canonical candidate a viable `Stage J` artifact, rather than only a structurally well-described export?

Comparison target:

- compare `artifacts/stage_j_qwen_paper_consistent/` against `artifacts/stage_j_qwen_redesign/`

Minimum pass conditions for this round:

- the regression must run successfully on the canonical candidate;
- the candidate must materially improve over the historical bridge baseline on the same core metrics;
- the candidate must not collapse to the historical bridge behavior where generation exact match is zero.

This round does **not** require token-perfect equality with the redesign source. It requires a non-degenerate, clearly improved correctness result that is sufficient to support **export-visible completion**.

Fail meaning:

- even if structural proofs pass, `Stage J` cannot yet be called export-visible complete.

## 7. Completion Status Model

At the end of the round, `Stage J` must resolve to one of two explicit statuses:

### 7.1 Export-visible complete

This status is allowed when all of the following hold:

- canonical candidate artifact exists;
- standard weight proof passes;
- attention export-visible proof passes;
- FFN export-visible proof passes;
- norm export-visible proof passes;
- correctness regression passes the round's acceptance bar.

Meaning:

- `Stage J` is complete at the level of export-visible evidence;
- `Stage K` may still remain pending.

### 7.2 Not complete

This status is mandatory if any one of the five evidence items fails or is missing.

Meaning:

- the repository must identify the blocking component explicitly;
- the result may still produce useful evidence, but must not be described as `Stage J` complete.

## 8. Execution Order

Implementation must follow this order:

1. Materialize the canonical candidate export chain
2. Generate the unified evidence package against the canonical candidate
3. Update tests so they validate the candidate and evidence package
4. Update canonical docs to center `Stage J` on the canonical candidate
5. Produce a final completion summary with pass/fail status

This order is intentional:

- without the candidate artifact, all later evidence remains tied to intermediate lines;
- without unified proofs, docs would over-claim completion;
- without updated tests, the repository would not guard the new acceptance target.

## 9. Documentation Changes

The canonical documents must be updated after implementation:

- `docs/阶段J_论文一致部署路线说明.md`
- `docs/论文一致最终部署主线.md`
- `README.md` if the canonical `Stage J` entry or active artifact wording changes

Required narrative changes:

- the new canonical candidate becomes the active `Stage J` target;
- the historical bridge becomes auxiliary evidence only;
- docs must state whether `Stage J` reached export-visible completion or remains blocked on a named component.

## 10. Testing Expectations

Tests must move from "bridge exists and reports exist" to "canonical candidate exists and unified completion evidence exists".

At minimum, tests must guard:

- canonical candidate artifact path exists or is produced by the export path under test;
- unified evidence package paths exist;
- completion summary contains explicit status fields;
- historical bridge is not treated as the active canonical `Stage J` answer in canonical docs;
- docs and tests agree on the same acceptance target.

## 11. Risks

### 11.1 Structural proof without viable behavior

Risk:

- the candidate may become structurally well-described but still behave too much like the historical bridge.

Mitigation:

- correctness regression is a mandatory evidence item.

### 11.2 Narrative drift

Risk:

- code may create a canonical candidate while docs continue describing intermediate lines as the active target.

Mitigation:

- canonical doc updates and doc tests are part of the same round.

### 11.3 False completion

Risk:

- a component proof may degrade into a narrative-only note without a clear pass/fail judgment.

Mitigation:

- each proof must yield an explicit pass/fail result and completion summary must aggregate them.

## 12. Success Criteria

This round is successful if the repository can honestly say:

> `Stage J` now has one canonical `paper-consistent` candidate artifact exported directly from the redesign source, together with a unified evidence package showing standard weight visibility, attention/FFN/norm export-visible proof, and a non-degenerate correctness regression.

This round is not successful if the best available statement is still:

> the historical bridge is loadable and some component audits exist.
