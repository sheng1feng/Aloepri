# Qwen Stage H-K Redesign Design

**Date:** 2026-04-14

## 1. Goal

Redefine `Stage H`, `Stage I`, `Stage J`, and `Stage K` for the Qwen line so they align with the AloePri paper's deployment-adaptation logic:

- deployment compatibility means preserving a standard Transformer runtime graph;
- deployment compatibility does **not** mean collapsing all obfuscation into one simple global hidden transform;
- complexity should stay in offline parameter construction whenever it can still be absorbed into standard component weights.

This redesign keeps the `H/I/J/K` stage names, but rewrites their meanings and closure criteria. The current `H/I/J/K` implementation history remains valuable, but it is reclassified as legacy evidence rather than the canonical target definition.

## 2. Problem Statement

The current Qwen deployment line drifted toward a conservative interpretation of "deployable":

- current `Stage H` focuses on research-line artifact stabilization, static attention, and noise calibration;
- current `Stage I` narrows to a `Stage A`-style standard HF/vLLM entrance;
- current `Stage J` centers on a non-expanding, standard-shape, square-monomial full-layer transform;
- current `Stage K` packages that conservative `Stage J` line for delivery.

That path succeeded in producing standard artifacts and a usable delivery line, but it also over-compressed the paper's deployment-capable obfuscation expression. Recent security results show that the resulting deployment line is substantially weaker than the research line under structure-recovery attacks.

The redesign therefore treats the current deployment line as a useful baseline and cautionary example, not as the final interpretation of AloePri deployment adaptation.

## 3. Scope

### In scope

- Re-define Qwen `Stage H/I/J/K`
- Reorganize the stage documentation hierarchy around the new semantics
- Add migration notes from old `H/I/J/K` to the new meaning
- Implement and verify new Qwen `Stage H/I/J/K` closure sequentially
- Partially resync security reporting at the total-report / board level after the new stage line is in place

### Out of scope

- Rewriting `Stage A-G`
- Full rewrite of all `docs/qwen_security/Gate*.md` documents in the same pass
- Llama-specific redesign during this pass
- Formal re-proof of the AloePri theorems or PAC bounds

## 4. Design Principles

### 4.1 Standard runtime graph, not standardized obfuscation

The server runtime should still look like a standard Transformer to HF / vLLM / SGLang. However, parameter expressions may remain complex as long as they can be compiled into standard component weights.

### 4.2 Offline complexity is preferred

If a perturbation can be encoded by offline parameter rewriting without introducing a custom online operator, it should remain a candidate deployment expression rather than being removed prematurely.

### 4.3 Internal-layer protection is first-class

Embedding/head protection alone is insufficient. Deployable expressions must explicitly consider attention, FFN, and norm-side structure preservation.

### 4.4 Stage closure must be complete

Each redesigned stage must close its own loop before the next stage starts:

- stage definition document
- code implementation
- tests
- runnable validation
- stage report
- explicit exit criteria

## 5. New Stage Semantics

## 5.1 Stage H: Deployable Obfuscation Expression Reconstruction

### Purpose

Recover the subset of paper-level obfuscation expression that is still deployable under a standard Transformer runtime.

### Core questions

- Which paper mechanisms can still be absorbed into standard Qwen component weights?
- Which mechanisms require approximation?
- Which mechanisms are truly incompatible with the intended deployment target?

### Expected focus

- attention-side deployable perturbation inventory
- FFN-side deployable perturbation inventory
- norm correction / fusion rules
- key-matrix retention boundary
- per-layer / per-head / per-block / per-group diversity policy

### Outputs

- a Qwen deployable-obfuscation specification
- component-level prototype builders or transforms
- Stage H validation outputs for component-level correctness
- a Stage H report documenting preserved, approximated, and rejected expressions

### Non-goals

- full-model export
- release packaging
- final deployment API

### Exit criteria

- the repository contains a canonical Stage H spec and report;
- Stage H code expresses paper-aligned deployable component rewrites, not only global square monomial adaptation;
- Stage H tests and validation outputs pass;
- the retained deployable perturbation set is explicit and reviewable.

## 5.2 Stage I: Deployment-Constraint Validation

### Purpose

Validate whether the Stage H expression set is actually compatible with the intended standard inference surfaces.

### Core questions

- Can the Stage H expressions be materialized into standard HF-compatible weights?
- Which expressions remain acceptable to vLLM/SGLang assumptions?
- Which compatibility constraints are hard blockers versus engineering tasks?

### Expected focus

- standard weight materialization constraints
- runtime contract for client/server secret placement
- HF loadability checks
- vLLM/SGLang compatibility matrix
- dtype and fusion boundary analysis

### Outputs

- deployment contract document
- compatibility matrix
- feasibility / blocker report
- Stage I validation scripts and tests

### Non-goals

- final full-model deployable checkpoint
- release packaging

### Exit criteria

- the intended deployable expression set has a validated compatibility story;
- blockers are classified, not guessed;
- Stage I validation scripts and tests pass on the supported environment.

## 5.3 Stage J: Qwen Full-Model Deployment Materialization

### Purpose

Materialize the validated Stage H expression set into a full Qwen deployable obfuscated checkpoint.

### Core questions

- Can the new deployment expression survive full-model writeback?
- Does the full model preserve correctness well enough to remain a serious deployment candidate?
- Does it retain more structural diversity than the old conservative standard-shape line?

### Expected focus

- full-layer parameter writeback
- full-model artifact generation
- correctness regression
- profile definition grounded in the new expression set

### Outputs

- new Stage J Qwen artifacts
- Stage J regression outputs
- Stage J report comparing the new line against the legacy conservative line

### Non-goals

- final release packaging
- security Gate reimplementation inside Stage J itself

### Exit criteria

- a full deployable checkpoint exists for the redesigned line;
- Stage J regression passes according to explicit thresholds;
- the report documents how the redesigned line differs from legacy `stage_j_full_square`.

## 5.4 Stage K: Release Packaging and Runtime Closure

### Purpose

Package the redesigned Stage J artifacts into a stable delivery surface.

### Core questions

- What are the supported profiles?
- What is the canonical release directory layout?
- What is the supported client/server inference contract?

### Expected focus

- release catalog and manifests
- profile naming
- inference entrypoints
- delivery docs

### Outputs

- new Stage K release artifact
- Stage K packaging report
- updated user-facing usage docs for the redesigned Qwen line

### Non-goals

- new obfuscation mathematics

### Exit criteria

- the release artifact is generated from the redesigned Stage J line;
- packaging tests and inference smoke checks pass;
- user-facing docs point to the new release path, not only the legacy one.

## 6. Legacy Mapping

The current repository history should not be discarded. Instead, it is remapped as follows:

- old `Stage H`: legacy research-line deployment-oriented artifact stabilization
- old `Stage I`: legacy standard-entry feasibility probe, centered on `Stage A`
- old `Stage J`: legacy conservative standard-shape deployment line
- old `Stage K`: legacy packaging of the conservative `Stage J` line

This implies:

- legacy artifacts remain valid as baselines and regression references;
- legacy reports remain useful but should stop serving as the canonical definition of redesigned `H/I/J/K`;
- new documents must make the distinction explicit.

## 7. Documentation Strategy

### 7.1 Canonical documents to create or rewrite

- new canonical `Stage H` design/report document
- new canonical `Stage I` deployment-validation document
- new canonical `Stage J` materialization report
- new canonical `Stage K` release report
- one migration overview document explaining old-vs-new stage meanings

### 7.2 Legacy documents to preserve with explicit labeling

The following current documents should be preserved but reclassified as legacy or archived references:

- `docs/阶段H-Attention静态化与噪声定标报告.md`
- `docs/阶段H_混淆模型部署说明.md`
- `docs/阶段I_vLLM接入计划.md`
- `docs/阶段I_vLLM复现报告.md`
- `docs/阶段J_标准形状前缀恢复报告.md`
- `docs/阶段J_标准形状噪声定标报告.md`
- `docs/阶段K_标准形状交付包装报告.md`

### 7.3 Security-document sync boundary

During this redesign pass:

- sync `docs/qwen_security/Qwen安全总报告.md` at the summary level after the redesigned line lands;
- sync `docs/qwen_security/推进看板.md` at the stage-status level;
- defer full Gate-level wording rewrites unless a Gate depends directly on the new artifacts.

## 8. Closure Model

Each stage must close in this order:

1. stage design / definition document
2. test-first implementation of the relevant code changes
3. stage-local validation scripts / outputs
4. stage report update
5. explicit verification run

No later stage may start implementation until the current stage reaches that closure point.

## 9. Planned Execution Order

1. establish new documentation and migration skeleton
2. close redesigned `Stage H`
3. close redesigned `Stage I`
4. close redesigned `Stage J`
5. close redesigned `Stage K`
6. update partial security summary / board references

## 10. Success Criteria

This redesign is successful if:

- `H/I/J/K` once again mean "paper-aligned deployable line" rather than "legacy conservative standard-shape line";
- the repository contains a full redesigned Qwen deployment path from expression reconstruction to packaged release;
- each stage has code, tests, docs, and validation evidence;
- the security narrative can clearly distinguish legacy deployment artifacts from the redesigned deployment artifacts.
