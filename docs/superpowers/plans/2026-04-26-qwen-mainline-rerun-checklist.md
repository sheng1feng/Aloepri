# Qwen Mainline Rerun Checklist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an executable Qwen-only rerun checklist to the canonical Qwen mainline document without creating any new active documentation entry.

**Architecture:** Extend the existing Qwen docs regression tests first so they require a dedicated rerun-checklist section and the new section numbering, then update the canonical markdown root to describe the active release surface, active security targets, runnable scripts, inherited Stage-J correctness limitation, and the missing Stage-K-native correctness evidence. Keep all remaining-rerun guidance inside `docs/论文一致最终部署主线.md`.

**Tech Stack:** Markdown, `pytest`, existing docs regression tests in `tests/test_qwen_paper_consistent_docs.py`

---

## File Map

- Modify: `tests/test_qwen_paper_consistent_docs.py`
  - Owns canonical-root regression coverage for Qwen documentation wording, section structure, and active artifact names.
- Modify: `docs/论文一致最终部署主线.md`
  - Owns the only active Qwen-wide deployment-line narrative and remaining-work summary.

### Task 1: Add The Executable Qwen Rerun Checklist To The Canonical Root

**Files:**
- Modify: `tests/test_qwen_paper_consistent_docs.py:71-154`
- Modify: `docs/论文一致最终部署主线.md:77-125`
- Test: `tests/test_qwen_paper_consistent_docs.py`

- [ ] **Step 1: Write the failing test contract for the new checklist section**

Replace the section-number regexes and add the new checklist assertion in `tests/test_qwen_paper_consistent_docs.py`:

```python
def test_mainline_doc_next_steps_are_real_remaining_work() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 8\. 下一步顺序(.*?)## 9\.", text, re.S)
    assert section is not None
    next_steps = section.group(1)

    assert "唯一 release 面" in next_steps
    assert "correctness" in next_steps
    assert "`VMA / IMA / ISA`" in next_steps
    assert "`default` / `reference`" in next_steps

    assert "固定唯一主线文档与阶段主报告" not in next_steps
    assert "合并 Stage-J 冗余文档" not in next_steps
    assert "移除 legacy / bridge / redesign 并列叙事" not in next_steps


def test_mainline_doc_no_longer_lists_stage_k_cutover_as_remaining_work() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 8\. 下一步顺序(.*?)## 9\.", text, re.S)
    assert section is not None
    next_steps = section.group(1)
    assert "唯一 release 面" in next_steps
    assert "correctness" in next_steps
    assert "`VMA / IMA / ISA`" in next_steps
    assert "将 `Stage K` 切换到最终论文一致线" not in next_steps


def test_qwen_mainline_doc_has_explicit_paper_gap_section() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    assert "## 7. 与原始论文的当前差异" in text
    assert "不能表述为“已经与论文完全等价”" in text
    assert "`paper_consistent`" in text
    assert "`VMA / IMA / ISA`" in text


def test_qwen_mainline_doc_lists_executable_rerun_checklist() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 6\. 剩余复跑可执行清单(.*?)## 7\.", text, re.S)
    assert section is not None
    checklist = section.group(1)

    assert "`artifacts/stage_k_release`" in checklist
    assert "`default`" in checklist
    assert "`reference`" in checklist
    assert "`stage_k_default`" in checklist
    assert "`stage_k_reference`" in checklist
    assert "`scripts/export_stage_k_release.py`" in checklist
    assert "`scripts/infer_stage_k_release.py`" in checklist
    assert "`scripts/security_qwen/run_vma.py`" in checklist
    assert "`scripts/security_qwen/run_ima.py`" in checklist
    assert "`scripts/security_qwen/run_isa.py`" in checklist
    assert "`outputs/stage_j/paper_consistent/correctness_regression.json`" in checklist
    assert "当前 correctness 证据仍继承自 `Stage J`" in checklist
    assert "还缺 `Stage K` 自身的 correctness 结果文件" in checklist
    assert "`outputs/security_qwen/vma/stage_k_default.json`" in checklist
    assert "`outputs/security_qwen/ima/stage_k_reference.json`" in checklist
    assert "`outputs/security_qwen/isa/stage_k_default.hidden_state.json`" in checklist
    assert "`hidden_state`" in checklist
    assert "`attention_score`" in checklist
    assert "不再是当前主线复跑对象" in checklist
```

- [ ] **Step 2: Run the focused docs tests and verify the new assertions fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q \
  tests/test_qwen_paper_consistent_docs.py::test_qwen_mainline_doc_lists_executable_rerun_checklist \
  tests/test_qwen_paper_consistent_docs.py::test_mainline_doc_next_steps_are_real_remaining_work \
  tests/test_qwen_paper_consistent_docs.py::test_mainline_doc_no_longer_lists_stage_k_cutover_as_remaining_work \
  tests/test_qwen_paper_consistent_docs.py::test_qwen_mainline_doc_has_explicit_paper_gap_section
```

Expected:

- FAIL because `## 6. 剩余复跑可执行清单` does not exist yet
- FAIL because the old section numbers still point at `## 6 / ## 7 / ## 8`

- [ ] **Step 3: Write the minimal canonical-root markdown to satisfy the new contract**

In `docs/论文一致最终部署主线.md`, insert the new rerun section after `## 5. 当前证据入口`, then renumber the later sections from `6/7/8/9` to `7/8/9/10`.

Use this exact new section body:

```md
## 6. 剩余复跑可执行清单

以下清单只对应当前唯一活跃 Qwen 发布面：

- release surface：`artifacts/stage_k_release`
- active profiles：`default` / `reference`
- active security targets：`stage_k_default` / `stage_k_reference`

以下对象不再是当前主线复跑对象：

- buffered redesign
- standard-visible bridge
- 历史 `Stage J` deployment targets，例如 `stage_j_stable_reference`、`stage_j_tiny_a`

### 6.1 release-surface correctness

当前 correctness 证据仍继承自 `Stage J`：

- `outputs/stage_j/paper_consistent/correctness_regression.json`

因此，这一轮还缺的不是再解释一次 `Stage J`，而是补出 `Stage K` 自身的 correctness 结果文件。

当前可直接执行的最小检查入口是：

1. 用 `scripts/export_stage_k_release.py` 重新导出或确认 `artifacts/stage_k_release`
2. 用 `scripts/infer_stage_k_release.py` 分别对 `default` / `reference` 做 release-surface smoke

这一步只能证明当前 release profile 仍然可运行、可见、可推理；它还不能代替最终 release-surface correctness 证据。

当前主线还缺的收口证据应落在 `Stage K` 自身路径下，例如：

- `outputs/stage_k_release/correctness/default.json`
- `outputs/stage_k_release/correctness/reference.json`

或等价的单一汇总文件：

- `outputs/stage_k_release/correctness_summary.json`

### 6.2 `VMA / IMA / ISA` 复跑

当前主线只认 `stage_k_default` 与 `stage_k_reference` 两个安全目标。

`VMA`：

- 执行入口：`scripts/security_qwen/run_vma.py`
- 预期结果文件：
  - `outputs/security_qwen/vma/stage_k_default.json`
  - `outputs/security_qwen/vma/stage_k_reference.json`

`IMA`：

- 执行入口：`scripts/security_qwen/run_ima.py`
- 预期结果文件：
  - `outputs/security_qwen/ima/stage_k_default.json`
  - `outputs/security_qwen/ima/stage_k_reference.json`

`ISA`：

- 执行入口：`scripts/security_qwen/run_isa.py`
- 结果必须显式记录 observable 类型
- 当前脚本支持：
  - `hidden_state`
  - `attention_score`
- 例如：
  - `outputs/security_qwen/isa/stage_k_default.hidden_state.json`
  - `outputs/security_qwen/isa/stage_k_reference.hidden_state.json`

如果需要统一比较视图，可以再用 comparison/export 脚本做汇总；但主线完成判定仍以每个 active target 的单独结果文件为准。

### 6.3 完成判定

只有同时满足下面几项，Qwen 当前唯一 release 面才算完成这轮主线复跑：

1. `artifacts/stage_k_release` 已重新导出或明确复核
2. `default` / `reference` 都已完成 release-surface smoke
3. `stage_k_default` / `stage_k_reference` 的 `VMA` 结果已落盘
4. `stage_k_default` / `stage_k_reference` 的 `IMA` 结果已落盘
5. `stage_k_default` / `stage_k_reference` 的 `ISA` 结果已按所选 observable 落盘
6. 主线文档可以直接引用这些 `Stage K` release-surface 结果，而不是继续借 `Stage J` inherited evidence 或历史 security 文档代指
```

Then renumber the existing headings exactly like this:

```md
## 7. 与原始论文的当前差异
## 8. 下一步顺序
## 9. 历史与中间证据说明
## 10. 历史文档去向
```

- [ ] **Step 4: Run the focused docs tests and verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q \
  tests/test_qwen_paper_consistent_docs.py::test_qwen_mainline_doc_lists_executable_rerun_checklist \
  tests/test_qwen_paper_consistent_docs.py::test_mainline_doc_next_steps_are_real_remaining_work \
  tests/test_qwen_paper_consistent_docs.py::test_mainline_doc_no_longer_lists_stage_k_cutover_as_remaining_work \
  tests/test_qwen_paper_consistent_docs.py::test_qwen_mainline_doc_has_explicit_paper_gap_section
```

Expected:

- PASS

- [ ] **Step 5: Run the broader docs/release regression suite**

Run:

```bash
conda run --no-capture-output -n qwen-transformers pytest -q \
  tests/test_mainline_docs_history.py \
  tests/test_qwen_paper_consistent_docs.py \
  tests/test_stage_hk_alignment_checklist.py \
  tests/test_security_qwen_summary.py \
  tests/test_stage_k_release.py \
  tests/test_stage_k_llama_release.py
git diff --check
```

Expected:

- all listed tests PASS
- `git diff --check` prints nothing

- [ ] **Step 6: Commit only the intended Qwen doc/test changes**

Run:

```bash
git add tests/test_qwen_paper_consistent_docs.py docs/论文一致最终部署主线.md
git commit -m "docs: add qwen rerun checklist to mainline root"
```
