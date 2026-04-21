# Qwen Paper-Consistent Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redirect Stage J work from bridge-equivalence tuning toward a direct paper-consistent deployment artifact for Qwen.

**Architecture:** Preserve the current buffered redesign and standard-visible bridge as references, but stop treating either as the final target. The next implementation line should directly construct a standard-visible deployment artifact that retains paper-relevant attention / FFN / norm expression as much as possible.

**Tech Stack:** Python, PyTorch, Transformers, safetensors, existing `src/stage_h.py`, `src/stage_j_standard_bridge.py`, `src/security_qwen/*`, pytest.

---

### Task 1: Add paper-consistent deployment target descriptor

**Files:**
- Create: `src/stage_j_paper_consistent.py`
- Create: `tests/test_stage_j_paper_consistent.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage_j_paper_consistent import build_stage_j_paper_consistent_target


def test_stage_j_paper_consistent_target_marks_bridge_as_non_final() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["bridge_is_final_target"] is False
    assert payload["standard_graph_required"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

- [ ] **Step 3: Write minimal implementation**

Create the target descriptor with fields:

- `standard_graph_required = True`
- `standard_visible_keys_required = True`
- `bridge_is_final_target = False`
- `buffered_reference = artifacts/stage_j_qwen_redesign`

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

- [ ] **Step 5: Commit**

```bash
git add src/stage_j_paper_consistent.py tests/test_stage_j_paper_consistent.py
git commit -m "feat: define paper-consistent stage j target"
```

### Task 2: Add export-visible component gap report

**Files:**
- Create: `src/stage_j_component_gap.py`
- Create: `scripts/run_stage_j_component_gap.py`
- Create: `tests/test_stage_j_component_gap.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage_j_component_gap import build_stage_j_component_gap_report


def test_component_gap_report_flags_norm_as_unresolved() -> None:
    payload = build_stage_j_component_gap_report()
    assert payload["norm"]["status"] == "unresolved"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_component_gap.py
```

- [ ] **Step 3: Write minimal implementation**

Make the report enumerate:

- `embed/head`
- `attention`
- `ffn`
- `norm`

with `status` and `notes`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_component_gap.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_j_component_gap.py
```

- [ ] **Step 5: Commit**

```bash
git add src/stage_j_component_gap.py scripts/run_stage_j_component_gap.py tests/test_stage_j_component_gap.py
git commit -m "feat: add stage j component gap report"
```
