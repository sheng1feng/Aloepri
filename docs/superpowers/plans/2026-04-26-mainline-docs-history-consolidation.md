# Mainline Docs And History Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate active Qwen and Llama documentation into explicit mainline roots, move non-mainline documents into `docs/history/`, and update tests plus README so only the intended active roots remain visible.

**Architecture:** Keep `docs/` root as a deliberately small active surface containing only the repository root, one Qwen root, one Llama root, their retained support documents, and paper-reference sources. Move historical material into a structured `docs/history/{qwen,llama,security,shared}/` tree, then update tests and entry points to validate the new routing model. Because the current baseline already contains a Llama `Stage K` field-name regression, fix that first so the branch can return to a green baseline before the larger documentation migration.

**Tech Stack:** Markdown documentation, pytest document tests, git worktrees, shell file moves, Python dataclass-based release metadata.

---

### Task 1: Repair the Current Llama Stage-K Metadata Regression

**Files:**
- Modify: `src/stage_k_llama_release.py`
- Test: `tests/test_stage_k_llama_release.py`

- [ ] **Step 1: Confirm the existing Llama release test fails for the current field name mismatch**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_stage_k_llama_release.py
```

Expected: FAIL with `TypeError: StageKProfile.__init__() got an unexpected keyword argument 'regression_file'`.

- [ ] **Step 2: Update the Llama Stage-K profile builder to use the new correctness-evidence field**

Replace the profile entries in `src/stage_k_llama_release.py` so they match the current `StageKProfile` shape:

```python
def default_stage_k_llama_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="stable_reference",
            source_dir="artifacts/stage_j_llama_real_full_square",
            description="Zero-noise Llama-3.2-3B standard-shape full-layer checkpoint.",
            recommended_use="Regression baseline, correctness debugging, deterministic demos.",
            correctness_evidence_file="outputs/stage_j_llama/real_remote_validation.json",
        ),
        StageKProfile(
            name="tiny_a",
            source_dir="artifacts/stage_j_llama_real_full_square_tiny_a",
            description="Recommended non-zero noise Llama-3.2-3B standard-shape full-layer checkpoint.",
            recommended_use="Default delivery profile for Llama-3.2-3B when non-zero obfuscation noise is required.",
            correctness_evidence_file="outputs/stage_j_llama/real_tiny_a_remote_validation.json",
        ),
    ]
```

- [ ] **Step 3: Re-run the focused Llama release test**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_stage_k_llama_release.py
```

Expected: PASS.

- [ ] **Step 4: Commit the baseline fix**

Run:

```bash
git add src/stage_k_llama_release.py tests/test_stage_k_llama_release.py
git commit -m "fix: align llama stage k correctness metadata"
```

Expected: one commit containing only the Llama metadata compatibility fix.

### Task 2: Add Failing Tests for the New Active Root Layout

**Files:**
- Create: `tests/test_mainline_docs_history.py`
- Modify: `tests/test_qwen_paper_consistent_docs.py`
- Modify: `README.md`

- [ ] **Step 1: Write a new failing test file for the repository/Qwen/Llama root layout and history migration**

Create `tests/test_mainline_docs_history.py` with these tests:

```python
from pathlib import Path


def test_active_root_docs_exist() -> None:
    assert Path("docs/复现主线总览.md").exists()
    assert Path("docs/论文一致最终部署主线.md").exists()
    assert Path("docs/Llama-3.2-3B最终部署主线.md").exists()


def test_history_subtrees_exist() -> None:
    for path in [
        "docs/history/qwen",
        "docs/history/llama",
        "docs/history/security",
        "docs/history/shared",
    ]:
        assert Path(path).is_dir()


def test_qwen_security_tree_moves_under_history() -> None:
    assert not Path("docs/qwen_security").exists()
    assert Path("docs/history/security/qwen_security/README.md").exists()


def test_old_shared_entry_docs_move_under_history() -> None:
    assert not Path("docs/仓库技术文档.md").exists()
    assert not Path("docs/Qwen与Llama复现阶段区分说明.md").exists()
    assert Path("docs/history/shared/仓库技术文档.md").exists()
    assert Path("docs/history/shared/Qwen与Llama复现阶段区分说明.md").exists()


def test_old_llama_entry_docs_move_under_history() -> None:
    assert not Path("docs/Llama-3.2-3B快速使用说明.md").exists()
    assert not Path("docs/Llama-3.2-3B云端验证说明.md").exists()
    assert not Path("docs/Llama-3.2-3B本机改造与云验证计划.md").exists()
    assert Path("docs/history/llama/Llama-3.2-3B快速使用说明.md").exists()
    assert Path("docs/history/llama/Llama-3.2-3B云端验证说明.md").exists()
    assert Path("docs/history/llama/Llama-3.2-3B本机改造与云验证计划.md").exists()
```

- [ ] **Step 2: Extend the existing Qwen docs test to assert the new active entry set**

Append these assertions to `tests/test_qwen_paper_consistent_docs.py`:

```python
def test_readme_points_to_repository_and_dual_mainlines() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "docs/复现主线总览.md" in text
    assert "docs/论文一致最终部署主线.md" in text
    assert "docs/Llama-3.2-3B最终部署主线.md" in text


def test_llama_active_docs_are_rooted_under_llama_mainline() -> None:
    text = Path("docs/Llama-3.2-3B最终部署主线.md").read_text(encoding="utf-8")
    assert "docs/Llama-3.2-3B标准形状恢复报告.md" in text
    assert "docs/Llama-3.2-3B噪声定标与StageK推进说明.md" in text
    assert "docs/Llama-3.2-3B客户端与Server使用说明.md" in text
    assert "docs/history/llama/Llama-3.2-3B快速使用说明.md" in text


def test_security_docs_move_to_history_tree() -> None:
    text = Path("docs/复现主线总览.md").read_text(encoding="utf-8")
    assert "docs/history/security/qwen_security/" in text
```

- [ ] **Step 3: Run the new and updated tests to verify they fail for the missing root/history layout**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py
```

Expected: FAIL because `docs/复现主线总览.md`, `docs/Llama-3.2-3B最终部署主线.md`, and `docs/history/*` do not exist yet, and `README.md` still points to old active entry sets.

- [ ] **Step 4: Commit the red tests**

Run:

```bash
git add tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py
git commit -m "test: capture mainline docs and history layout"
```

Expected: commit containing only the new failing expectations.

### Task 3: Create the Repository Root and Llama Root Mainline Documents

**Files:**
- Create: `docs/复现主线总览.md`
- Create: `docs/Llama-3.2-3B最终部署主线.md`
- Modify: `docs/Llama-3.2-3B标准形状恢复报告.md`
- Modify: `docs/Llama-3.2-3B噪声定标与StageK推进说明.md`
- Modify: `docs/Llama-3.2-3B客户端与Server使用说明.md`

- [ ] **Step 1: Write the repository-level active root**

Create `docs/复现主线总览.md` with sections covering:

```markdown
# 复现主线总览

## 1. 当前仓库只保留两条活跃主线
- Qwen：论文一致最终部署线
- Llama-3.2-3B：最终部署主线

## 2. 整个复现流程摘要
- Stage A-D：baseline 与早期恢复链路
- Stage E-G：复杂 attention、KeyMat 与结构表达恢复
- Stage H-I：可部署混淆表达与部署约束验证
- Stage J-K：标准可交付工件与 release 包装

## 3. Qwen 当前主线
- 入口：`docs/论文一致最终部署主线.md`
- 活跃阶段文档：H / I / J / K
- 当前唯一 release：`artifacts/stage_k_release`

## 4. Llama 当前主线
- 入口：`docs/Llama-3.2-3B最终部署主线.md`
- 活跃支撑文档：标准形状恢复、噪声定标与 Stage K、client/server 使用说明
- 当前唯一 release：`artifacts/stage_k_llama_release`

## 5. 当前部署线与原始论文的差异
- 已对齐项
- 已可部署但仍非论文完全同态的项
- 仍需补证或仍弱于论文理想的项

## 6. 历史文档去向
- `docs/history/qwen/`
- `docs/history/llama/`
- `docs/history/security/`
- `docs/history/shared/`
```

- [ ] **Step 2: Write the active Llama root**

Create `docs/Llama-3.2-3B最终部署主线.md` with sections covering:

```markdown
# Llama-3.2-3B 最终部署主线

## 1. 唯一目标
- 以 `artifacts/stage_k_llama_release` 作为当前唯一活跃交付面

## 2. 当前状态
- adapter 接入
- 本机 smoke 与标准形状导出链路成立
- 真实 4090 correctness 验证完成
- 噪声定标与 Stage K release 完成

## 3. 当前 release 语义
- `stable_reference`
- `tiny_a`

## 4. 活跃文档
- `docs/Llama-3.2-3B标准形状恢复报告.md`
- `docs/Llama-3.2-3B噪声定标与StageK推进说明.md`
- `docs/Llama-3.2-3B客户端与Server使用说明.md`

## 5. 与 Qwen 的差异
- Qwen 以 `paper_consistent` 为根
- Llama 仍沿用自身 release profile 语义

## 6. 与论文的差异
- 哪些点已对齐
- 哪些点仍是工程近似/证据化而非完全论文同态

## 7. 历史文档
- `docs/history/llama/...`
```

- [ ] **Step 3: Add canonical notes to the retained active Llama support documents**

Prepend this canonical note block to each retained active Llama support document:

```markdown
> Canonical note: 本文档只回答当前 `Llama-3.2-3B` 某一局部主题，不承担全局主线说明。Llama 唯一主线入口见 [docs/Llama-3.2-3B最终部署主线.md](Llama-3.2-3B最终部署主线.md)。
```

Apply it to:

- `docs/Llama-3.2-3B标准形状恢复报告.md`
- `docs/Llama-3.2-3B噪声定标与StageK推进说明.md`
- `docs/Llama-3.2-3B客户端与Server使用说明.md`

- [ ] **Step 4: Run the focused tests after adding the new root docs**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py
```

Expected: still FAIL because the history migration and README cleanup are not done yet, but failures should now move past the missing-root-doc stage.

- [ ] **Step 5: Commit the new active roots**

Run:

```bash
git add docs/复现主线总览.md docs/Llama-3.2-3B最终部署主线.md docs/Llama-3.2-3B标准形状恢复报告.md docs/Llama-3.2-3B噪声定标与StageK推进说明.md docs/Llama-3.2-3B客户端与Server使用说明.md
git commit -m "docs: add repository and llama mainline roots"
```

Expected: commit containing the new root docs and Llama canonical-note updates.

### Task 4: Recenter README and the Active Qwen Root Set

**Files:**
- Modify: `README.md`
- Modify: `docs/论文一致最终部署主线.md`
- Modify: `docs/阶段H_Qwen可部署混淆表达重构报告.md`
- Modify: `docs/阶段I_部署约束验证报告.md`
- Modify: `docs/阶段J_论文一致部署路线说明.md`
- Modify: `docs/阶段K_Qwen交付包装报告.md`
- Modify: `tests/test_qwen_paper_consistent_docs.py`

- [ ] **Step 1: Rewrite README so only the repository root plus Qwen/Llama roots act as active entry points**

Update `README.md` so it:

- starts by pointing first to `docs/复现主线总览.md`;
- presents Qwen active root as `docs/论文一致最终部署主线.md`;
- presents Llama active root as `docs/Llama-3.2-3B最终部署主线.md`;
- stops listing `docs/qwen_security/*`, `docs/仓库技术文档.md`, `docs/Qwen与Llama复现阶段区分说明.md`, `docs/Llama-3.2-3B快速使用说明.md`, and `docs/Llama-3.2-3B云端验证说明.md` as active top-level entry points.

Use this active-entry block shape:

```markdown
## 活跃文档入口

- 仓库总入口：[`docs/复现主线总览.md`](docs/复现主线总览.md)
- Qwen 主线：[`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)
- Llama 主线：[`docs/Llama-3.2-3B最终部署主线.md`](docs/Llama-3.2-3B最终部署主线.md)
```

- [ ] **Step 2: Add history-routing notes to the active Qwen root and retained stage docs**

Append a concise section to `docs/论文一致最终部署主线.md`:

```markdown
## 8. 历史文档去向

- Qwen 旧阶段/旧审计文档：`docs/history/qwen/`
- 安全子域旧文档：`docs/history/security/qwen_security/`
- 跨线旧总览文档：`docs/history/shared/`
```

Preserve the existing canonical notes in the retained Qwen stage docs, but add one sentence in each if needed making clear that historical evidence has moved under `docs/history/`.

- [ ] **Step 3: Update the active docs test expectations to the new README/mainline shape**

Adjust `tests/test_qwen_paper_consistent_docs.py` so it asserts:

- `README.md` contains `docs/复现主线总览.md`;
- `README.md` contains `docs/Llama-3.2-3B最终部署主线.md`;
- `README.md` no longer treats `docs/qwen_security/README.md` or old Llama quick/cloud docs as active top-level entry points;
- `docs/论文一致最终部署主线.md` contains the history routing section.
- the old `test_security_docs_are_subordinate_to_main_line()` logic now reads from `docs/history/security/qwen_security/README.md` and `docs/history/security/qwen_security/推进看板.md`.

- [ ] **Step 4: Run the focused tests again**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py
```

Expected: still FAIL only on unmigrated historical paths, not on missing active-root/README expectations.

- [ ] **Step 5: Commit the active-entry cleanup**

Run:

```bash
git add README.md docs/论文一致最终部署主线.md docs/阶段H_Qwen可部署混淆表达重构报告.md docs/阶段I_部署约束验证报告.md docs/阶段J_论文一致部署路线说明.md docs/阶段K_Qwen交付包装报告.md tests/test_qwen_paper_consistent_docs.py
git commit -m "docs: recenter active documentation roots"
```

Expected: commit containing the active README/Qwen-root cleanup only.

### Task 5: Move Historical Documents Into `docs/history/` And Update Links

**Files:**
- Create: `docs/history/qwen/`
- Create: `docs/history/llama/`
- Create: `docs/history/security/`
- Create: `docs/history/shared/`
- Move: the historical files listed in the approved spec
- Modify: any active root/support docs that still reference old root-level historical paths
- Test: `tests/test_mainline_docs_history.py`

- [ ] **Step 1: Create the history directory structure**

Run:

```bash
mkdir -p docs/history/qwen docs/history/llama docs/history/security docs/history/shared
```

Expected: the four history subtrees exist.

- [ ] **Step 2: Move the Qwen historical documents**

Run:

```bash
mv docs/阶段A_B严格复现报告.md docs/history/qwen/
mv docs/阶段C_block0完整恢复报告.md docs/history/qwen/
mv docs/阶段E复杂Attention复现报告.md docs/history/qwen/
mv docs/阶段E排错与修正报告.md docs/history/qwen/
mv docs/阶段F-KeyMat复现计划与结果.md docs/history/qwen/
mv docs/阶段G-KeyMat融合化报告.md docs/history/qwen/
mv docs/阶段H-Attention静态化与噪声定标报告.md docs/history/qwen/
mv docs/阶段H_混淆模型部署说明.md docs/history/qwen/
mv docs/阶段I_Phase2_最小原型报告.md docs/history/qwen/
mv docs/阶段I_Phase2_非扩维可逆变换设计.md docs/history/qwen/
mv docs/阶段I_vLLM复现报告.md docs/history/qwen/
mv docs/阶段I_vLLM接入计划.md docs/history/qwen/
mv docs/阶段J_Attention缺口审计报告.md docs/history/qwen/
mv docs/阶段J_KeyMat候选搜索报告.md docs/history/qwen/
mv docs/阶段J_KeyMat网格搜索报告.md docs/history/qwen/
mv docs/阶段J_Norm缺口审计报告.md docs/history/qwen/
mv docs/阶段J_标准形状前缀恢复报告.md docs/history/qwen/
mv docs/阶段J_标准形状协变恢复计划.md docs/history/qwen/
mv docs/阶段J_标准形状噪声定标报告.md docs/history/qwen/
mv docs/阶段J_桥接等价性回归报告.md docs/history/qwen/
mv docs/阶段K_标准形状交付包装报告.md docs/history/qwen/
mv docs/阶段H-K论文对齐检查表.md docs/history/qwen/
mv docs/阶段H-K重构迁移说明.md docs/history/qwen/
```

Expected: those files no longer exist at `docs/` root.

- [ ] **Step 3: Move the Llama historical documents**

Run:

```bash
mv docs/Llama-3.2-3B快速使用说明.md docs/history/llama/
mv docs/Llama-3.2-3B云端验证说明.md docs/history/llama/
mv docs/Llama-3.2-3B本机改造与云验证计划.md docs/history/llama/
```

Expected: only the retained Llama mainline root/support docs remain in `docs/` root.

- [ ] **Step 4: Move the shared and security historical documents**

Run:

```bash
mv docs/Qwen与Llama复现阶段区分说明.md docs/history/shared/
mv docs/仓库技术文档.md docs/history/shared/
mv docs/阶段A-K模块化整理报告.md docs/history/shared/
mv docs/完整复现总报告_阶段A-D.md docs/history/shared/
mv docs/完整复现总报告_阶段A-E.md docs/history/shared/
mv docs/完整复现总报告_阶段A-F.md docs/history/shared/
mv docs/完整复现总报告_阶段A-G.md docs/history/shared/
mv docs/完整复现总报告_阶段A-H.md docs/history/shared/
mv docs/完整复现总报告_阶段A-K.md docs/history/shared/
mv docs/qwen_security docs/history/security/
```

Expected: `docs/qwen_security/` disappears and `docs/history/security/qwen_security/` exists.

- [ ] **Step 5: Update active links to the new history paths**

Search for stale root-level historical references:

```bash
rg -n "docs/qwen_security|docs/仓库技术文档|docs/Qwen与Llama复现阶段区分说明|docs/Llama-3.2-3B快速使用说明|docs/Llama-3.2-3B云端验证说明|docs/阶段H-K论文对齐检查表|docs/阶段J_Attention缺口审计报告" README.md docs tests
```

Then edit the active root/support documents and tests so any intentional historical reference points to the new `docs/history/...` path.

- [ ] **Step 6: Run the focused documentation tests**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py
```

Expected: PASS.

- [ ] **Step 7: Commit the history migration**

Run:

```bash
git add docs/history README.md docs/复现主线总览.md docs/论文一致最终部署主线.md docs/Llama-3.2-3B最终部署主线.md docs/Llama-3.2-3B标准形状恢复报告.md docs/Llama-3.2-3B噪声定标与StageK推进说明.md docs/Llama-3.2-3B客户端与Server使用说明.md tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py
git commit -m "docs: move non-mainline docs into history"
```

Expected: commit containing the bulk history move plus active-link fixes.

### Task 6: Final Verification And Branch Wrap-Up

**Files:**
- Verify only; no new files required

- [ ] **Step 1: Run the full focused verification set**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q tests/test_mainline_docs_history.py tests/test_qwen_paper_consistent_docs.py tests/test_stage_k_release.py tests/test_stage_k_llama_release.py
```

Expected: PASS.

- [ ] **Step 2: Check active-entry grep output**

Run:

```bash
rg -n "docs/qwen_security/README.md|docs/Llama-3.2-3B快速使用说明.md|docs/Llama-3.2-3B云端验证说明.md|docs/仓库技术文档.md|docs/Qwen与Llama复现阶段区分说明.md" README.md docs/复现主线总览.md docs/论文一致最终部署主线.md docs/Llama-3.2-3B最终部署主线.md
```

Expected: no matches, or only matches that already use the new `docs/history/...` paths.

- [ ] **Step 3: Check formatting safety**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Commit any final verification-driven touchups**

Run:

```bash
git status --short
```

Expected: clean working tree. If verification forced a tiny follow-up edit, stage it and commit with:

```bash
git add <changed-files>
git commit -m "docs: finalize mainline history consolidation"
```

- [ ] **Step 5: Prepare branch-completion handoff**

Run:

```bash
git log --oneline --decorate -n 6
```

Expected: a concise sequence of commits for:

- the Llama metadata baseline fix;
- red tests for the new docs architecture;
- new active root docs;
- active-entry cleanup;
- history migration;
- any tiny final verification fix if needed.
