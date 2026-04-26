# Mainline Docs And History Consolidation Design

## 1. Context

The repository documentation tree currently mixes:

- active Qwen deployment-line documents;
- active Llama deployment-line documents;
- old Qwen stage reports and transitional audits;
- old Llama plans and usage notes;
- security-side evidence and planning material;
- historical summary reports and repository handoff notes.

This creates two practical problems:

1. the current canonical deployment lines are harder to find than they should be;
2. older evidence and transitional reports still behave like semi-active entry points.

The user wants the documentation tree to be reduced to active mainline documents only, while preserving the historical corpus under a dedicated `docs/history/` tree.

The user also chose these boundaries:

- keep the original paper files and core paper-reference notes in `docs/` root;
- keep one active Qwen mainline, with the existing canonical root plus a small stage report set;
- keep one active Llama mainline, structured similarly with a main root plus a small active document set;
- move old and non-mainline material into `docs/history/`;
- keep `docs/superpowers/` untouched because it is workflow metadata, not reproduction-line documentation.

## 2. Goals

This consolidation must:

1. create a single repository-level documentation entry for the current reproduction lines;
2. preserve one active Qwen deployment line and one active Llama deployment line;
3. summarize the full `Stage A -> K` reproduction flow in one active root document;
4. explicitly document the current gap between the deployed lines and the original paper ideal;
5. move non-mainline documentation into a structured `docs/history/` tree;
6. update active entry points and tests so historical documents no longer behave like current primary references.

## 3. Non-Goals

This consolidation does not:

- change model code, artifacts, evaluations, or release semantics;
- re-run security or correctness experiments;
- rewrite every historical document to match the new terminology;
- move `docs/superpowers/` plans/specs into `history`;
- remove the original paper PDF/TXT or the active paper-reference notes from `docs/` root.

## 4. Canonical Active Information Architecture

After consolidation, `docs/` root must contain only:

### 4.1 Repository-level active roots

- `docs/复现主线总览.md`
- `docs/论文一致最终部署主线.md`
- `docs/Llama-3.2-3B最终部署主线.md`

### 4.2 Active Qwen mainline stage reports

- `docs/阶段H_Qwen可部署混淆表达重构报告.md`
- `docs/阶段I_部署约束验证报告.md`
- `docs/阶段J_论文一致部署路线说明.md`
- `docs/阶段K_Qwen交付包装报告.md`

### 4.3 Active Llama mainline support documents

- `docs/Llama-3.2-3B标准形状恢复报告.md`
- `docs/Llama-3.2-3B噪声定标与StageK推进说明.md`
- `docs/Llama-3.2-3B客户端与Server使用说明.md`

### 4.4 Active paper/reference sources

- `docs/AloePri 论文中的部署适配机制整理.md`
- `docs/AloePri_技术报告梳理与复现方案.md`
- `docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf`
- `docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).txt`

Everything else outside `docs/superpowers/` must move under `docs/history/`.

## 5. Active Root Responsibilities

### 5.1 `docs/复现主线总览.md`

This becomes the repository's top-level active documentation root.

It must answer:

- what the full `Stage A -> K` reproduction flow accomplished;
- what the active Qwen deployment line is;
- what the active Llama deployment line is;
- which active documents define each line;
- how the current deployment lines differ from the original paper ideal;
- where historical evidence and old reports now live.

It must be the only active document that summarizes both Qwen and Llama together at the repository level.

### 5.2 `docs/论文一致最终部署主线.md`

This remains the only active Qwen root.

It must continue to answer:

- Qwen current target definition;
- Qwen active `Stage H/I/J/K` meanings;
- current Qwen completion state;
- remaining Qwen work;
- active Qwen evidence entry points.

It must add a concise section that explains where historical Qwen/security/shared documents moved under `docs/history/`.

### 5.3 `docs/Llama-3.2-3B最终部署主线.md`

This becomes the only active Llama root.

It must answer:

- current active Llama deployment-line definition;
- active release surface: `artifacts/stage_k_llama_release`;
- current profile semantics: `stable_reference` and `tiny_a`;
- active Llama document set;
- major milestones from adapter enablement through Stage K release;
- current gap versus the original paper ideal;
- remaining Llama work;
- where historical Llama documents moved under `docs/history/llama/`.

## 6. Required Mainline Content

### 6.1 Full reproduction-flow summary

`docs/复现主线总览.md` must contain a concise end-to-end narrative of the repository’s current reproduction flow:

- `Stage A-D`: baseline and early restoration chain;
- `Stage E-G`: complex attention, KeyMat, and structural expression work;
- `Stage H-I`: deployment-compatible expression and deployability validation;
- `Stage J-K`: standard-visible deployment artifact and release packaging.

This is a process summary, not a per-file changelog.

### 6.2 Deployment-line vs. paper difference summary

`docs/复现主线总览.md` must include a direct, explicit comparison between the current deployment lines and the original paper target.

It should distinguish at least:

- what is already aligned;
- what is operationally deployable but not fully paper-ideal;
- what remains weaker, simplified, or unproven relative to the paper.

The wording must be concrete and avoid pretending the current lines are identical to the paper if they are not.

### 6.3 Qwen/Llama contrast

The active roots must make clear:

- Qwen is currently organized around the paper-consistent final deployment line;
- Llama follows the same high-level reproduction logic but still uses its own release semantics and evidence shape;
- Qwen and Llama are parallel active lines, not one merged line.

## 7. Active Document Routing Rules

To prevent new ambiguity, active documents must follow these roles.

### 7.1 Repository root role

Only `docs/复现主线总览.md` can summarize both active lines together and discuss the overall reproduction flow across Qwen and Llama.

### 7.2 Qwen root role

Only `docs/论文一致最终部署主线.md` can define the active Qwen deployment-line state across stages.

### 7.3 Llama root role

Only `docs/Llama-3.2-3B最终部署主线.md` can define the active Llama deployment-line state across stages.

### 7.4 Stage/support role

The retained stage/support documents may explain stage-local or usage-local details, but they must not present themselves as alternative global roots.

Each retained active non-root document should carry a short canonical note pointing to its owning root.

## 8. History Tree Design

`docs/history/` must be structured so the historical corpus stays navigable instead of becoming one flat dump.

### 8.1 Required subtrees

- `docs/history/qwen/`
- `docs/history/llama/`
- `docs/history/security/`
- `docs/history/shared/`

### 8.2 `docs/history/qwen/`

This subtree holds old Qwen-specific stage reports, audits, migration notes, and transitional documents that are no longer active entry points.

Include:

- old Stage A-G reports;
- old Stage H/I/J/K reports not retained in the active set;
- Qwen-specific audits such as attention/norm/keymat/bridge reports;
- migration and alignment notes that are still useful historically but not active roots.

### 8.3 `docs/history/llama/`

This subtree holds Llama documents that remain useful historically but are no longer part of the active line.

Include at minimum:

- `docs/Llama-3.2-3B快速使用说明.md`
- `docs/Llama-3.2-3B云端验证说明.md`
- `docs/Llama-3.2-3B本机改造与云验证计划.md`

### 8.4 `docs/history/security/`

Move the current `docs/qwen_security/` tree here as-is, under:

- `docs/history/security/qwen_security/`

This keeps security evidence intact while removing it from the active top-level documentation surface.

### 8.5 `docs/history/shared/`

This subtree holds cross-line or repository-wide historical reports and handoff material.

Include:

- `docs/Qwen与Llama复现阶段区分说明.md`
- `docs/仓库技术文档.md`
- `docs/阶段A-K模块化整理报告.md`
- `docs/完整复现总报告_阶段A-D.md`
- `docs/完整复现总报告_阶段A-E.md`
- `docs/完整复现总报告_阶段A-F.md`
- `docs/完整复现总报告_阶段A-G.md`
- `docs/完整复现总报告_阶段A-H.md`
- `docs/完整复现总报告_阶段A-K.md`

## 9. File Treatment Matrix

### 9.1 Keep active in `docs/` root

Keep:

- `docs/论文一致最终部署主线.md`
- `docs/阶段H_Qwen可部署混淆表达重构报告.md`
- `docs/阶段I_部署约束验证报告.md`
- `docs/阶段J_论文一致部署路线说明.md`
- `docs/阶段K_Qwen交付包装报告.md`
- `docs/Llama-3.2-3B标准形状恢复报告.md`
- `docs/Llama-3.2-3B噪声定标与StageK推进说明.md`
- `docs/Llama-3.2-3B客户端与Server使用说明.md`
- the four active paper/reference files listed in section 4.4

Create:

- `docs/复现主线总览.md`
- `docs/Llama-3.2-3B最终部署主线.md`

### 9.2 Move to `docs/history/qwen/`

Move:

- `docs/阶段A_B严格复现报告.md`
- `docs/阶段C_block0完整恢复报告.md`
- `docs/阶段E复杂Attention复现报告.md`
- `docs/阶段E排错与修正报告.md`
- `docs/阶段F-KeyMat复现计划与结果.md`
- `docs/阶段G-KeyMat融合化报告.md`
- `docs/阶段H-Attention静态化与噪声定标报告.md`
- `docs/阶段H_混淆模型部署说明.md`
- `docs/阶段I_Phase2_最小原型报告.md`
- `docs/阶段I_Phase2_非扩维可逆变换设计.md`
- `docs/阶段I_vLLM复现报告.md`
- `docs/阶段I_vLLM接入计划.md`
- `docs/阶段J_Attention缺口审计报告.md`
- `docs/阶段J_KeyMat候选搜索报告.md`
- `docs/阶段J_KeyMat网格搜索报告.md`
- `docs/阶段J_Norm缺口审计报告.md`
- `docs/阶段J_标准形状前缀恢复报告.md`
- `docs/阶段J_标准形状协变恢复计划.md`
- `docs/阶段J_标准形状噪声定标报告.md`
- `docs/阶段J_桥接等价性回归报告.md`
- `docs/阶段K_标准形状交付包装报告.md`
- `docs/阶段H-K论文对齐检查表.md`
- `docs/阶段H-K重构迁移说明.md`

### 9.3 Move to `docs/history/llama/`

Move:

- `docs/Llama-3.2-3B快速使用说明.md`
- `docs/Llama-3.2-3B云端验证说明.md`
- `docs/Llama-3.2-3B本机改造与云验证计划.md`

### 9.4 Move to `docs/history/security/`

Move:

- `docs/qwen_security/` -> `docs/history/security/qwen_security/`

### 9.5 Move to `docs/history/shared/`

Move:

- `docs/Qwen与Llama复现阶段区分说明.md`
- `docs/仓库技术文档.md`
- `docs/阶段A-K模块化整理报告.md`
- `docs/完整复现总报告_阶段A-D.md`
- `docs/完整复现总报告_阶段A-E.md`
- `docs/完整复现总报告_阶段A-F.md`
- `docs/完整复现总报告_阶段A-G.md`
- `docs/完整复现总报告_阶段A-H.md`
- `docs/完整复现总报告_阶段A-K.md`

## 10. README And Entry-Point Cleanup

`README.md` must be rewritten so that active documentation entry points are limited to:

- `docs/复现主线总览.md`
- `docs/论文一致最终部署主线.md`
- `docs/Llama-3.2-3B最终部署主线.md`

It may still mention the retained Qwen and Llama support documents, but only under their respective active roots.

It must stop exposing:

- `docs/qwen_security/*` as active top-level entry points;
- old Qwen reports in root `docs/`;
- old Llama quick/plan/cloud documents as active entry points;
- historical shared reports as current documentation roots.

## 11. Links And Notes

Implementation must update active-document links so that:

- retained active documents link to the new repository/Qwen/Llama roots correctly;
- any remaining references to moved documents use their new `docs/history/...` paths;
- no active root links to a historical document as if it were an active peer root.

For retained Llama support documents, add canonical notes that point to:

- `docs/Llama-3.2-3B最终部署主线.md`

For retained Qwen stage documents, preserve the existing pattern of canonical notes pointing to:

- `docs/论文一致最终部署主线.md`

## 12. Verification

Completion of this consolidation requires:

1. document tests updated for the new active-root and history-path layout;
2. active roots and retained support documents present at the expected paths;
3. moved historical documents no longer present in `docs/` root;
4. `README.md` exposing only the intended active documentation roots;
5. link checks via focused repository grep for old active-entry references;
6. `git diff --check` clean.

## 13. Out Of Scope Follow-Ups

This round does not require:

- rewriting the internal body of every historical document after migration;
- consolidating `docs/superpowers/`;
- changing active Llama profile semantics to mirror Qwen;
- inventing a new security-document taxonomy beyond moving the current tree under `docs/history/security/`.
