# Qwen Paper-Consistent Documentation Consolidation Design

## 1. Context

The repository currently exposes multiple overlapping Qwen `Stage H/I/J/K` narratives:

- legacy conservative `standard-shape` reports;
- redesigned buffered deployment-line reports;
- standard-visible bridge reports;
- security-side summary documents that partially restate deployment status.

This makes the current documentation tree hard to trust because more than one document can appear to answer the same question:

> what is the current canonical Qwen deployment line, and what is still missing before it becomes the final paper-consistent deliverable line?

The user wants to remove that ambiguity and keep only one progress line:

> the paper-consistent final deployable line.

The user also chose a `evidence-preserving` cleanup strategy:

- keep a small set of supporting documents as evidence;
- stop allowing them to behave like parallel progress lines.

## 2. Goals

This consolidation must:

1. create a single canonical documentation entry for the Qwen paper-consistent deployment line;
2. reduce Qwen `Stage H/I/J/K` documentation to one progress narrative;
3. keep exactly one canonical stage report per stage;
4. preserve key evidence documents without letting them compete with the main line;
5. remove redundant entry points from `README.md` and other index-style docs.

## 3. Non-Goals

This consolidation does not:

- change model code, artifacts, or evaluation outputs;
- claim that the paper-consistent deployment artifact already exists;
- rerun correctness or security evaluation;
- archive the full historical corpus into a new `archive/` tree in this pass.

## 4. Canonical Information Architecture

After consolidation, Qwen deployment documentation must follow exactly this hierarchy:

### 4.1 Canonical root

Create a single canonical root document:

- `docs/论文一致最终部署主线.md`

This document becomes the only place allowed to answer:

- what the final target is;
- where current progress stands;
- what `Stage H/I/J/K` mean now;
- what remains blocked;
- what the next execution order is.

### 4.2 Canonical stage reports

Keep one canonical stage report per stage:

- `docs/阶段H_Qwen可部署混淆表达重构报告.md`
- `docs/阶段I_部署约束验证报告.md`
- `docs/阶段J_论文一致部署路线说明.md`
- `docs/阶段K_Qwen交付包装报告.md`

These four documents remain stage-specific references, but they must no longer present themselves as independent competing roadmaps.

### 4.3 Evidence documents

Keep a small evidence set, but demote it from roadmap status:

- `docs/AloePri 论文中的部署适配机制整理.md`
- `docs/阶段H-K论文对齐检查表.md`
- `docs/qwen_security/Qwen安全总报告.md`
- `docs/qwen_security/部署线重构与当前安全简报.md`
- `docs/阶段J_Attention缺口审计报告.md`
- `docs/阶段J_Norm缺口审计报告.md`
- `docs/阶段J_桥接等价性回归报告.md`
- `docs/阶段J_KeyMat候选搜索报告.md`
- `docs/阶段J_KeyMat网格搜索报告.md`

These documents may provide proof, measurements, or reasoning, but they must not be treated as current progress hubs.

### 4.4 Historical documents

Legacy or transitional documents may remain on disk if they still carry historical value, but they must lose all main-entry visibility.

They must not appear in:

- `README.md` as current Qwen progress references;
- the new canonical root document as peer roadmaps;
- Qwen stage summaries as alternative active lines.

## 5. Document Routing Rules

To keep only one progress line, every Qwen deployment document must fit one of four roles.

### 5.1 Role A: canonical progress line

Only `docs/论文一致最终部署主线.md` can:

- summarize current state across stages;
- define the canonical next steps;
- describe what is still missing overall;
- explain how legacy, redesign, and bridge artifacts relate to the final target.

### 5.2 Role B: stage-specific canonical state

The four canonical stage reports may only answer:

- what this stage is supposed to do;
- what current evidence shows for this stage;
- what stage-local blockers remain.

They must not reintroduce a separate global roadmap.

### 5.3 Role C: evidence

Evidence documents may answer:

- why a statement is believed;
- what audit or regression data supports it;
- what gap was measured.

They must not act as progress trackers.

### 5.4 Role D: history

Historical documents may answer:

- what an older line attempted;
- what earlier terminology meant;
- what the old baseline accomplished.

They must not appear as active stage references.

## 6. File Treatment Matrix

### 6.1 Keep as canonical stage reports

Keep these unchanged in role, but update wording if needed so they clearly behave as current stage references:

- `docs/阶段H_Qwen可部署混淆表达重构报告.md`
- `docs/阶段I_部署约束验证报告.md`
- `docs/阶段J_论文一致部署路线说明.md`
- `docs/阶段K_Qwen交付包装报告.md`

### 6.2 Keep as evidence documents

Keep these files, but remove them from top-level progress navigation:

- `docs/AloePri 论文中的部署适配机制整理.md`
- `docs/阶段H-K论文对齐检查表.md`
- `docs/qwen_security/Qwen安全总报告.md`
- `docs/qwen_security/部署线重构与当前安全简报.md`
- `docs/阶段J_Attention缺口审计报告.md`
- `docs/阶段J_Norm缺口审计报告.md`
- `docs/阶段J_桥接等价性回归报告.md`
- `docs/阶段J_KeyMat候选搜索报告.md`
- `docs/阶段J_KeyMat网格搜索报告.md`

### 6.3 Merge into canonical Stage J narrative, then remove standalone entry status

These documents all serve the same Stage-J question and should be absorbed into the canonical `Stage J` report:

- `docs/阶段J_Qwen全模型部署物化报告.md`
- `docs/阶段J_标准可见桥接导出报告.md`
- `docs/阶段J_标准权重证明报告.md`

Their key conclusions should be merged into `docs/阶段J_论文一致部署路线说明.md`.

After merge, these three files must be deleted.

Before deletion, all important conclusions must be copied into `docs/阶段J_论文一致部署路线说明.md`, and all repository references to the deleted files must be updated.

### 6.4 Demote to historical / legacy references

These documents describe legacy, transitional, or prototype lines and should no longer appear in active Qwen deployment entry points:

- `docs/阶段H-Attention静态化与噪声定标报告.md`
- `docs/阶段H_混淆模型部署说明.md`
- `docs/阶段I_vLLM复现报告.md`
- `docs/阶段I_vLLM接入计划.md`
- `docs/阶段I_Phase2_最小原型报告.md`
- `docs/阶段I_Phase2_非扩维可逆变换设计.md`
- `docs/阶段J_标准形状前缀恢复报告.md`
- `docs/阶段J_标准形状噪声定标报告.md`
- `docs/阶段J_标准形状协变恢复计划.md`
- `docs/阶段K_标准形状交付包装报告.md`
- `docs/阶段H-K重构迁移说明.md`

These files stay on disk in this pass as historical references, but they must stop appearing as current-line docs.

## 7. Canonical Root Document Content

The new `docs/论文一致最终部署主线.md` must contain these sections:

1. target definition
2. current single-line status summary
3. canonical `Stage H/I/J/K` meanings
4. current completion state by stage
5. what evidence supports the current state
6. what is still missing before final paper-consistent delivery
7. ordered next steps
8. evidence and history appendix

The appendix is the only place where legacy, redesign, and bridge lines may still be named together, and even there they must be described as:

- baseline evidence;
- intermediate evidence;
- export-path evidence.

They must not be described as parallel active roadmaps.

## 8. Entry-Point Cleanup

### 8.1 README

`README.md` must be rewritten so that Qwen deployment documentation exposes only one top-level entry:

- `docs/论文一致最终部署主线.md`

The README may still list the four canonical stage reports and a small evidence subset, but only under the main line.

It must stop presenting:

- legacy `Stage H-K`;
- redesigned buffered line;
- standard-visible bridge;
- standard-shape reports

as parallel Qwen progress surfaces.

### 8.2 Security index

`docs/qwen_security/README.md` stays as a security-subdomain index, but it must not function as the primary deployment-progress entry.

### 8.3 Security board

`docs/qwen_security/推进看板.md` stays on disk as a security-specific board only.

It must not continue to imply that it is the master Qwen deployment progress tracker.

## 9. Migration Strategy

Implementation should proceed in this order:

1. create `docs/论文一致最终部署主线.md`;
2. rewrite `README.md` to point to the new canonical root;
3. update the four canonical stage reports so their scope is stage-local and not a competing roadmap;
4. merge the three redundant Stage-J documents into `docs/阶段J_论文一致部署路线说明.md`;
5. remove top-level entry references to demoted historical docs;
6. check for broken references and adjust remaining links;
7. update `docs/qwen_security/README.md` and related index docs so security remains a subdomain, not the main deployment line.

## 10. Risks and Mitigations

### 10.1 Broken links

Risk:

- removing or merging Stage-J docs can leave stale references in `README.md`, other docs, or tests.

Mitigation:

- search all references before deletion;
- prefer direct content merge before file removal;
- only delete files after reference cleanup.

### 10.2 Accidental evidence loss

Risk:

- useful audit conclusions could disappear if “cleanup” is interpreted too aggressively.

Mitigation:

- preserve the chosen evidence set explicitly;
- ensure every removed roadmap doc has its important conclusions copied into either the canonical root or the canonical stage report.

### 10.3 Stage-J ambiguity survives

Risk:

- if Stage-J bridge and buffered redesign are still documented as peers, the repository will still look multi-track.

Mitigation:

- require `Stage J` canonical doc to describe both only as intermediate evidence on the path to the final paper-consistent line.

## 11. Acceptance Criteria

This design is complete when:

1. `docs/论文一致最终部署主线.md` exists and is the sole canonical Qwen deployment progress entry;
2. `README.md` points to one Qwen deployment main line, not multiple;
3. only four canonical Qwen `Stage H/I/J/K` reports remain as stage-level current docs;
4. Stage-J materialization / bridge / standard-weight proof no longer appear as three separate active progress documents;
5. selected evidence docs remain available but are clearly subordinate to the canonical line;
6. legacy and standard-shape docs no longer appear as active Qwen deployment references.
