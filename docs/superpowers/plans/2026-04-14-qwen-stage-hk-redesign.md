# Qwen Stage H-K Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Qwen `Stage H/I/J/K` line around paper-aligned deployment adaptation, then close each stage sequentially with docs, code, tests, and verification.

**Architecture:** Keep `Stage A-G` untouched, preserve legacy `H/I/J/K` artifacts as references, and layer a new canonical `H/I/J/K` line on top. Stage H defines deployable obfuscation expression inventory, Stage I validates deployment constraints, Stage J materializes a redesigned Qwen deployment artifact, and Stage K packages that artifact into a release surface.

**Tech Stack:** Python, PyTorch, Transformers, pytest, existing `src/stage_*` modules, JSON outputs under `outputs/`, reports under `docs/`.

---

### Task 1: Add Redesign Migration Scaffold

**Files:**
- Create: `docs/阶段H-K重构迁移说明.md`
- Modify: `README.md`
- Test: `tests/test_stage_hk_redesign_docs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_stage_hk_redesign_docs_exist() -> None:
    assert Path("docs/阶段H-K重构迁移说明.md").exists()


def test_readme_mentions_redesigned_qwen_stage_hk_line() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Qwen Stage H-K redesign" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_hk_redesign_docs.py
```

Expected: FAIL because the migration document and README text do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `docs/阶段H-K重构迁移说明.md` with:

```md
# 阶段 H-K 重构迁移说明

## 目标

说明旧版 `Stage H/I/J/K` 与新版 `Stage H/I/J/K` 的关系。

## 旧版映射

- 旧 `Stage H`：legacy research-line deployment-oriented artifact stabilization
- 旧 `Stage I`：legacy standard-entry feasibility probe
- 旧 `Stage J`：legacy conservative standard-shape deployment line
- 旧 `Stage K`：legacy packaging of legacy Stage J

## 新版主线

- 新 `Stage H`：可部署混淆表达重构
- 新 `Stage I`：部署约束验证
- 新 `Stage J`：Qwen 全模型部署物化
- 新 `Stage K`：交付包装与运行时收口
```

Patch `README.md` to include a short section:

```md
## Qwen Stage H-K redesign

The repository now distinguishes:

- legacy conservative `Stage H/I/J/K` artifacts
- redesigned paper-aligned Qwen `Stage H/I/J/K`
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_hk_redesign_docs.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/阶段H-K重构迁移说明.md tests/test_stage_hk_redesign_docs.py
git commit -m "docs: add qwen stage h-k migration scaffold"
```

### Task 2: Close Redesigned Stage H

**Files:**
- Create: `docs/阶段H_Qwen可部署混淆表达重构报告.md`
- Create: `scripts/run_stage_h_deployable_inventory.py`
- Create: `tests/test_stage_h_redesign.py`
- Modify: `src/stage_h.py`
- Modify: `src/stage_h_attention_static.py`
- Modify: `docs/阶段H-Attention静态化与噪声定标报告.md`

- [ ] **Step 1: Write the failing test**

Add `tests/test_stage_h_redesign.py`:

```python
from src.stage_h import build_stage_h_deployable_inventory


def test_stage_h_deployable_inventory_marks_attention_diversity() -> None:
    payload = build_stage_h_deployable_inventory()
    assert payload["stage"] == "H"
    assert payload["attention"]["preserve_block_permutation"] is True
    assert payload["attention"]["preserve_head_or_group_diversity"] is True


def test_stage_h_deployable_inventory_marks_norm_correction() -> None:
    payload = build_stage_h_deployable_inventory()
    assert payload["norm"]["preserve_kappa_correction"] is True


def test_stage_h_deployable_inventory_has_legacy_note() -> None:
    payload = build_stage_h_deployable_inventory()
    assert "legacy_stage_h_scope" in payload["migration"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_h_redesign.py
```

Expected: FAIL because `build_stage_h_deployable_inventory` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `src/stage_h.py`:

```python
def build_stage_h_deployable_inventory() -> dict[str, object]:
    return {
        "stage": "H",
        "goal": "deployable_obfuscation_expression_reconstruction",
        "attention": {
            "preserve_block_permutation": True,
            "preserve_head_or_group_diversity": True,
            "preserve_rope_side_rotation_and_scaling": True,
            "current_builder": "build_staticized_attention",
        },
        "ffn": {
            "preserve_component_specific_transform": True,
            "current_builder": "build_keymat_fused_ffn",
        },
        "norm": {
            "preserve_kappa_correction": True,
            "current_builder": "build_keymat_fused_rmsnorm",
        },
        "migration": {
            "legacy_stage_h_scope": "attention_staticization_and_noise_calibration",
        },
    }
```

Create `scripts/run_stage_h_deployable_inventory.py`:

```python
from pathlib import Path
import json

from src.stage_h import build_stage_h_deployable_inventory


def main() -> None:
    payload = build_stage_h_deployable_inventory()
    out = Path("outputs/stage_h/deployable_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

Write `docs/阶段H_Qwen可部署混淆表达重构报告.md` with:

```md
# 阶段 H：Qwen 可部署混淆表达重构报告

## 新阶段目标

重新整理论文中仍可吸收到标准组件权重中的复杂扰动。

## 明确保留

- attention block/head/group diversity
- FFN component-specific perturbation
- norm-side kappa correction

## legacy 对照

旧版 Stage H 更偏 attention 静态化与噪声定标。
```

Patch `docs/阶段H-Attention静态化与噪声定标报告.md` header to label it as a legacy Stage H report.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_h.py tests/test_stage_h_redesign.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_h_deployable_inventory.py
```

Expected:

- pytest: PASS
- script: writes `outputs/stage_h/deployable_inventory.json`

- [ ] **Step 5: Commit**

```bash
git add src/stage_h.py src/stage_h_attention_static.py scripts/run_stage_h_deployable_inventory.py tests/test_stage_h_redesign.py docs/阶段H_Qwen可部署混淆表达重构报告.md docs/阶段H-Attention静态化与噪声定标报告.md outputs/stage_h/deployable_inventory.json
git commit -m "feat: close redesigned stage h inventory"
```

### Task 3: Close Redesigned Stage I

**Files:**
- Create: `docs/阶段I_部署约束验证报告.md`
- Create: `tests/test_stage_i_deployability.py`
- Modify: `src/stage_i_vllm.py`
- Modify: `scripts/run_stage_i_phase2_feasibility.py`
- Modify: `docs/阶段I_vLLM复现报告.md`

- [ ] **Step 1: Write the failing test**

Add `tests/test_stage_i_deployability.py`:

```python
from src.stage_i_vllm import build_stage_i_deployability_matrix


def test_stage_i_deployability_matrix_references_stage_h_inventory() -> None:
    payload = build_stage_i_deployability_matrix()
    assert payload["stage"] == "I"
    assert payload["source_stage"] == "H"


def test_stage_i_deployability_matrix_tracks_standard_runtime_boundary() -> None:
    payload = build_stage_i_deployability_matrix()
    assert payload["runtime_boundary"]["standard_transformer_graph"] is True
    assert "attention_diversity" in payload["validated_components"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_i_deployability.py
```

Expected: FAIL because `build_stage_i_deployability_matrix` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `src/stage_i_vllm.py`:

```python
from src.stage_h import build_stage_h_deployable_inventory


def build_stage_i_deployability_matrix() -> dict[str, Any]:
    inventory = build_stage_h_deployable_inventory()
    return {
        "stage": "I",
        "source_stage": "H",
        "runtime_boundary": {
            "standard_transformer_graph": True,
            "custom_online_operator_required": False,
        },
        "validated_components": {
            "attention_diversity": "needs_materialization_check",
            "ffn_component_transform": "needs_materialization_check",
            "norm_kappa_correction": "supported_if_fused_offline",
        },
        "inventory_goal": inventory["goal"],
    }
```

Update `scripts/run_stage_i_phase2_feasibility.py` to export the new matrix alongside the legacy feasibility summary.

Write `docs/阶段I_部署约束验证报告.md` with sections for:

- runtime boundary
- HF compatibility
- vLLM/SGLang compatibility
- blocker classification

Label `docs/阶段I_vLLM复现报告.md` as legacy Stage I.

- [ ] **Step 4: Run tests and validation**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_i.py tests/test_stage_i_deployability.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_i_phase2_feasibility.py
```

Expected:

- pytest: PASS
- script: updates `outputs/stage_i/phase2_feasibility.json`

- [ ] **Step 5: Commit**

```bash
git add src/stage_i_vllm.py scripts/run_stage_i_phase2_feasibility.py tests/test_stage_i_deployability.py docs/阶段I_部署约束验证报告.md docs/阶段I_vLLM复现报告.md outputs/stage_i/phase2_feasibility.json
git commit -m "feat: close redesigned stage i validation"
```

### Task 4: Close Redesigned Stage J

**Files:**
- Create: `docs/阶段J_Qwen全模型部署物化报告.md`
- Create: `src/stage_j_materialize.py`
- Create: `scripts/export_stage_j_redesign_checkpoint.py`
- Create: `scripts/run_stage_j_redesign_regression.py`
- Create: `tests/test_stage_j_redesign.py`
- Modify: `src/stage_j_block0.py`
- Modify: `docs/阶段J_标准形状前缀恢复报告.md`
- Modify: `docs/阶段J_标准形状噪声定标报告.md`

- [ ] **Step 1: Write the failing test**

Add `tests/test_stage_j_redesign.py`:

```python
from src.stage_j_materialize import build_stage_j_redesign_manifest


def test_stage_j_redesign_manifest_uses_stage_h_and_i_lineage() -> None:
    payload = build_stage_j_redesign_manifest()
    assert payload["stage"] == "J"
    assert payload["source_stages"] == ["H", "I"]


def test_stage_j_redesign_manifest_distinguishes_legacy_stage_j() -> None:
    payload = build_stage_j_redesign_manifest()
    assert payload["legacy_reference"] == "artifacts/stage_j_full_square"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_redesign.py
```

Expected: FAIL because `src.stage_j_materialize` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/stage_j_materialize.py`:

```python
from __future__ import annotations


def build_stage_j_redesign_manifest() -> dict[str, object]:
    return {
        "stage": "J",
        "goal": "qwen_full_model_deployment_materialization",
        "source_stages": ["H", "I"],
        "legacy_reference": "artifacts/stage_j_full_square",
        "redesign_export_dir": "artifacts/stage_j_qwen_redesign",
    }
```

Create `scripts/export_stage_j_redesign_checkpoint.py` and `scripts/run_stage_j_redesign_regression.py` as lightweight wrappers that emit manifest/regression JSON first, then can be extended with full export logic.

Write `docs/阶段J_Qwen全模型部署物化报告.md` describing:

- the redesigned artifact target
- lineage from Stage H and Stage I
- comparison against legacy `stage_j_full_square`

Label old Stage J docs as legacy conservative Stage J.

- [ ] **Step 4: Run tests and validation**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_block0.py tests/test_stage_j_redesign.py
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_redesign_checkpoint.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_j_redesign_regression.py
```

Expected:

- pytest: PASS
- scripts: write Stage J redesign JSON outputs

- [ ] **Step 5: Commit**

```bash
git add src/stage_j_materialize.py scripts/export_stage_j_redesign_checkpoint.py scripts/run_stage_j_redesign_regression.py tests/test_stage_j_redesign.py docs/阶段J_Qwen全模型部署物化报告.md docs/阶段J_标准形状前缀恢复报告.md docs/阶段J_标准形状噪声定标报告.md
git commit -m "feat: close redesigned stage j materialization"
```

### Task 5: Close Redesigned Stage K

**Files:**
- Create: `docs/阶段K_Qwen交付包装报告.md`
- Create: `tests/test_stage_k_redesign.py`
- Modify: `src/stage_k_release.py`
- Modify: `scripts/export_stage_k_release.py`
- Modify: `docs/阶段K_标准形状交付包装报告.md`

- [ ] **Step 1: Write the failing test**

Add `tests/test_stage_k_redesign.py`:

```python
from src.stage_k_release import default_stage_k_profiles


def test_stage_k_profiles_prefer_redesigned_stage_j_sources() -> None:
    profiles = default_stage_k_profiles()
    assert all("stage_j_qwen_redesign" in profile.source_dir for profile in profiles)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_k_redesign.py
```

Expected: FAIL because Stage K still points to legacy Stage J sources.

- [ ] **Step 3: Write minimal implementation**

Update `src/stage_k_release.py` profile defaults:

```python
StageKProfile(
    name="stable_reference",
    source_dir="artifacts/stage_j_qwen_redesign/stable_reference",
    ...
)
```

and:

```python
StageKProfile(
    name="tiny_a",
    source_dir="artifacts/stage_j_qwen_redesign/tiny_a",
    ...
)
```

Update `scripts/export_stage_k_release.py` if necessary to point to redesigned Stage J outputs.

Write `docs/阶段K_Qwen交付包装报告.md` with:

- redesigned release artifact layout
- profile semantics
- runtime contract

Label the old Stage K report as legacy packaging of the conservative line.

- [ ] **Step 4: Run tests and validation**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_k_release.py tests/test_stage_k_redesign.py
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_k_release.py
```

Expected:

- pytest: PASS
- export script: writes redesigned `catalog.json`

- [ ] **Step 5: Commit**

```bash
git add src/stage_k_release.py scripts/export_stage_k_release.py tests/test_stage_k_redesign.py docs/阶段K_Qwen交付包装报告.md docs/阶段K_标准形状交付包装报告.md
git commit -m "feat: close redesigned stage k packaging"
```

### Task 6: Partial Security-Summary Sync and Final Verification

**Files:**
- Modify: `docs/qwen_security/Qwen安全总报告.md`
- Modify: `docs/qwen_security/推进看板.md`
- Modify: `tests/test_security_qwen_summary.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_security_qwen_summary.py`:

```python
from pathlib import Path


def test_qwen_security_total_report_mentions_legacy_conservative_deployment_line() -> None:
    text = Path("docs/qwen_security/Qwen安全总报告.md").read_text(encoding="utf-8")
    assert "legacy conservative deployment line" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_security_qwen_summary.py
```

Expected: FAIL because the wording does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Patch the summary docs to:

- distinguish legacy `Stage J/K` security results from redesigned `Stage J/K`;
- update the board to note redesign progress and partial sync status.

- [ ] **Step 4: Run full verification**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q \
  tests/test_stage_h.py \
  tests/test_stage_h_redesign.py \
  tests/test_stage_i.py \
  tests/test_stage_i_deployability.py \
  tests/test_stage_j_block0.py \
  tests/test_stage_j_redesign.py \
  tests/test_stage_k_release.py \
  tests/test_stage_k_redesign.py \
  tests/test_security_qwen_summary.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/qwen_security/Qwen安全总报告.md docs/qwen_security/推进看板.md tests/test_security_qwen_summary.py
git commit -m "docs: sync security summary with qwen stage h-k redesign"
```
