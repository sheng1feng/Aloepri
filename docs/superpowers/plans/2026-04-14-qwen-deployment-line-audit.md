# Qwen Deployment-Line Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit whether the redesigned Qwen deployment line actually preserves paper-allowed deployable obfuscation expression in shipped artifacts and whether that expression improves the legacy deployment line's security weaknesses.

**Architecture:** Treat the current redesigned `Stage J/K` line as a bootstrap deployment line that must be audited before security reruns. First build artifact-level and deployability-boundary audits, then add redesigned security targets, and only then re-run `VMA -> IMA -> ISA` in that order.

**Tech Stack:** Python, JSON artifact inspection, existing `src/stage_h.py`, `src/stage_i_vllm.py`, `src/stage_j_materialize.py`, `src/stage_k_release.py`, `src/security_qwen/*`, pytest, existing security scripts.

---

### Task 1: Artifact-Level Expression Audit

**Files:**
- Create: `src/stage_hk_audit.py`
- Create: `scripts/run_stage_hk_expression_audit.py`
- Create: `tests/test_stage_hk_audit.py`
- Test: `tests/test_stage_hk_audit.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage_hk_audit import build_redesigned_expression_audit


def test_redesigned_expression_audit_detects_stage_h_bootstrap_expression() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_h_source"]["attention_profile_present"] is True
    assert payload["stage_h_source"]["keymat_parameters_present"] is True


def test_redesigned_expression_audit_marks_stage_j_as_bootstrap_not_full_proof() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_j"]["bootstraps_from_stage_h_pretrained"] is True
    assert payload["stage_j"]["has_component_level_expression_manifest"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_hk_audit.py
```

Expected: FAIL because `src.stage_hk_audit` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/stage_hk_audit.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_redesigned_expression_audit() -> dict[str, object]:
    stage_h_cfg = _load_json("artifacts/stage_h_pretrained/server/obfuscation_config.json")
    stage_j_manifest = _load_json("artifacts/stage_j_qwen_redesign/manifest.json")
    stage_k_catalog = _load_json("artifacts/stage_k_release/catalog.json")
    return {
        "stage_h_source": {
            "attention_profile_present": bool(stage_h_cfg.get("attention_profile")),
            "keymat_parameters_present": all(key in stage_h_cfg for key in ["lambda", "h"]),
        },
        "stage_j": {
            "bootstraps_from_stage_h_pretrained": stage_j_manifest.get("bootstrap_source") == "artifacts/stage_h_pretrained",
            "has_component_level_expression_manifest": "component_expression" in stage_j_manifest,
        },
        "stage_k": {
            "points_to_redesigned_stage_j": stage_k_catalog.get("stage_lineage") == "redesigned_qwen_stage_j",
        },
    }
```

Create `scripts/run_stage_hk_expression_audit.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_hk_audit import build_redesigned_expression_audit


def main() -> None:
    payload = build_redesigned_expression_audit()
    path = Path("outputs/stage_hk/redesign_expression_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_hk_audit.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_hk_expression_audit.py
```

Expected: PASS and `outputs/stage_hk/redesign_expression_audit.json` exists.

- [ ] **Step 5: Commit**

```bash
git add src/stage_hk_audit.py scripts/run_stage_hk_expression_audit.py tests/test_stage_hk_audit.py
git commit -m "feat: add redesigned deployment expression audit"
```

### Task 2: Deployability-Boundary Audit

**Files:**
- Create: `tests/test_stage_i_boundary_audit.py`
- Modify: `src/stage_i_vllm.py`
- Modify: `scripts/run_stage_i_phase2_feasibility.py`
- Test: `tests/test_stage_i_boundary_audit.py`

- [ ] **Step 1: Write the failing test**

```python
from src.stage_i_vllm import build_stage_i_boundary_audit


def test_stage_i_boundary_audit_tracks_standard_runtime_contract() -> None:
    payload = build_stage_i_boundary_audit()
    assert payload["runtime_graph_is_standard"] is True
    assert payload["custom_online_operator_required"] is False


def test_stage_i_boundary_audit_marks_expression_proof_gap() -> None:
    payload = build_stage_i_boundary_audit()
    assert payload["exported_artifact_proves_attention_expression"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_i_boundary_audit.py
```

Expected: FAIL because `build_stage_i_boundary_audit` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `src/stage_i_vllm.py`:

```python
def build_stage_i_boundary_audit() -> dict[str, object]:
    return {
        "runtime_graph_is_standard": True,
        "custom_online_operator_required": False,
        "exported_artifact_proves_attention_expression": False,
        "exported_artifact_proves_ffn_expression": False,
        "exported_artifact_proves_norm_expression": False,
    }
```

Also append this payload into `build_phase2_feasibility_summary()`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_i_boundary_audit.py tests/test_stage_i.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_i_phase2_feasibility.py
```

Expected: PASS and updated `outputs/stage_i/phase2_feasibility.json`.

- [ ] **Step 5: Commit**

```bash
git add src/stage_i_vllm.py scripts/run_stage_i_phase2_feasibility.py tests/test_stage_i_boundary_audit.py
git commit -m "feat: add redesigned deployability boundary audit"
```

### Task 3: Redesigned VMA Re-Attachment

**Files:**
- Modify: `src/security_qwen/artifacts.py`
- Create: `tests/test_security_qwen_redesign_targets.py`
- Create: `scripts/security_qwen/export_vma_redesign_comparison.py`
- Test: `tests/test_security_qwen_redesign_targets.py`

- [ ] **Step 1: Write the failing test**

```python
from src.security_qwen.artifacts import get_security_target, resolve_security_target


def test_redesign_stage_j_target_resolves() -> None:
    spec = get_security_target("stage_j_redesign")
    assert spec.artifact_dir == "artifacts/stage_j_qwen_redesign"


def test_redesign_stage_k_target_resolves() -> None:
    resolved = resolve_security_target("stage_k_redesign_tiny_a")
    assert resolved.resolved_root_dir.endswith("artifacts/stage_k_release/profiles/tiny_a")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_security_qwen_redesign_targets.py
```

Expected: FAIL because redesigned targets do not exist.

- [ ] **Step 3: Write minimal implementation**

Add redesigned target specs:

```python
SecurityTargetSpec(
    name="stage_j_redesign",
    stage="J",
    artifact_dir="artifacts/stage_j_qwen_redesign",
    profile=None,
    variant="redesigned_deployment_materialization",
    track="deployment_redesign",
    supported_attacks=["vma", "ima", "isa"],
)
```

and:

```python
SecurityTargetSpec(
    name="stage_k_redesign_tiny_a",
    stage="K",
    artifact_dir="artifacts/stage_k_release",
    profile="tiny_a",
    variant="redesigned_release_profile",
    track="release_redesign",
    supported_attacks=["vma", "ima", "isa"],
)
```

Add a comparison exporter that compares redesigned vs legacy `J/K`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_security_qwen_redesign_targets.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/security_qwen/artifacts.py tests/test_security_qwen_redesign_targets.py scripts/security_qwen/export_vma_redesign_comparison.py
git commit -m "feat: add redesigned security targets for vma"
```

### Task 4: Redesigned IMA Re-Attachment

**Files:**
- Create: `scripts/security_qwen/export_ima_redesign_comparison.py`
- Modify: `src/security_qwen/artifacts.py`
- Test: `tests/test_security_qwen_redesign_targets.py`

- [ ] **Step 1: Write the failing test**

```python
from src.security_qwen.artifacts import get_security_target


def test_redesign_targets_support_ima() -> None:
    assert "ima" in get_security_target("stage_j_redesign").supported_attacks
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_security_qwen_redesign_targets.py
```

Expected: FAIL before redesigned targets are complete.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/security_qwen/export_ima_redesign_comparison.py` that reuses the redesigned targets and writes:

```python
output_path = "outputs/security_qwen/summary/ima_redesign_comparison.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_security_qwen_redesign_targets.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/security_qwen/export_ima_redesign_comparison.py
git commit -m "feat: add redesigned ima comparison entrypoint"
```

### Task 5: Redesigned ISA Re-Attachment and Paper-Alignment Checklist

**Files:**
- Create: `docs/阶段H-K论文对齐检查表.md`
- Create: `scripts/security_qwen/export_isa_redesign_comparison.py`
- Create: `tests/test_stage_hk_alignment_checklist.py`
- Test: `tests/test_stage_hk_alignment_checklist.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_alignment_checklist_mentions_attention_and_norm() -> None:
    text = Path("docs/阶段H-K论文对齐检查表.md").read_text(encoding="utf-8")
    assert "attention" in text
    assert "norm" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_hk_alignment_checklist.py
```

Expected: FAIL because the checklist does not exist.

- [ ] **Step 3: Write minimal implementation**

Create checklist sections covering:

- embed/head noise + permutation + key matrix
- attention rotation / scaling / block perm / head-group diversity
- FFN transform retention
- norm kappa correction
- key-matrix expansion retention boundary

Create ISA exporter:

```python
output_path = "outputs/security_qwen/summary/isa_redesign_comparison.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_hk_alignment_checklist.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/阶段H-K论文对齐检查表.md scripts/security_qwen/export_isa_redesign_comparison.py tests/test_stage_hk_alignment_checklist.py
git commit -m "docs: add redesigned deployment paper-alignment checklist"
```
