# Qwen Stage K Paper-Consistent Release Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut over the active Qwen `Stage K` release surface to the single paper-consistent deployment line by making `artifacts/stage_k_release` package `artifacts/stage_j_qwen_paper_consistent` through `default` and `reference` profiles.

**Architecture:** Keep the existing `artifacts/stage_k_release` directory and export CLI, but replace the old redesign-oriented profile semantics with paper-consistent `default` / `reference` semantics. Update the security target registry and active docs to use the new `Stage K` naming, then regenerate the release artifact and verify the cutover end to end.

**Tech Stack:** Python, pytest, JSON catalog files, existing release export scripts, existing security target registry under `src/security_qwen`, Markdown docs.

---

### Task 1: Cut over Stage K profile defaults and catalog semantics

**Files:**
- Modify: `tests/test_stage_k_release.py`
- Modify: `tests/test_stage_k_redesign.py`
- Modify: `src/stage_k_llama_release.py`
- Modify: `src/stage_k_release.py`
- Modify: `scripts/infer_stage_k_release.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_stage_k_release.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.infer_stage_k_release import parse_args
from src.stage_k_release import StageKProfile, default_stage_k_profiles, export_stage_k_release


def _make_stage_k_source(source_dir: Path) -> None:
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")
    (source_dir / "manifest.json").write_text(
        json.dumps({"track": "paper_consistent_candidate"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_stage_k_profiles_use_default_and_reference() -> None:
    profiles = default_stage_k_profiles()
    assert [item.name for item in profiles] == ["default", "reference"]


def test_stage_k_profiles_point_to_paper_consistent_stage_j() -> None:
    profiles = default_stage_k_profiles()
    assert all(item.source_dir == "artifacts/stage_j_qwen_paper_consistent" for item in profiles)


def test_infer_stage_k_release_defaults_to_default_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["infer_stage_k_release.py", "--prompt", "hello"],
    )
    args = parse_args()
    assert args.profile == "default"


def test_export_stage_k_release_writes_paper_consistent_catalog(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_j_qwen_paper_consistent"
    _make_stage_k_source(source_dir)

    export_dir = tmp_path / "stage_k_release"
    catalog = export_stage_k_release(
        export_dir,
        profiles=[
            StageKProfile(
                name="default",
                source_dir=str(source_dir),
                description="Default paper-consistent delivery profile.",
                recommended_use="Default delivery entry for the paper-consistent release line.",
            ),
            StageKProfile(
                name="reference",
                source_dir=str(source_dir),
                description="Reference paper-consistent audit profile.",
                recommended_use="Audit and evidence entry for the same paper-consistent release line.",
            ),
        ],
        recommended_profile="default",
        reference_profile="reference",
        title="Stage-K Paper-Consistent Qwen Release",
    )

    assert catalog["stage_lineage"] == "paper_consistent_stage_j"
    assert catalog["recommended_profile"] == "default"
    assert catalog["reference_profile"] == "reference"
    assert [item["name"] for item in catalog["profiles"]] == ["default", "reference"]
```

Replace `tests/test_stage_k_redesign.py` with:

```python
from src.stage_k_release import default_stage_k_profiles


def test_stage_k_profiles_prefer_paper_consistent_stage_j_sources() -> None:
    profiles = default_stage_k_profiles()
    assert all("stage_j_qwen_paper_consistent" in profile.source_dir for profile in profiles)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_k_release.py tests/test_stage_k_redesign.py
```

Expected: FAIL because the current implementation still emits `stable_reference` / `tiny_a`, still points to `artifacts/stage_j_qwen_redesign`, and still uses `stable_reference_profile`.

- [ ] **Step 3: Write the minimal implementation**

Update `src/stage_k_release.py` by replacing `default_stage_k_profiles()` with:

```python
def default_stage_k_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="default",
            source_dir="artifacts/stage_j_qwen_paper_consistent",
            description="Default paper-consistent Stage-J Qwen release profile.",
            recommended_use="Default delivery entry for the paper-consistent Qwen deployment line.",
            regression_file="outputs/stage_j/paper_consistent/completion_summary.json",
        ),
        StageKProfile(
            name="reference",
            source_dir="artifacts/stage_j_qwen_paper_consistent",
            description="Reference paper-consistent Stage-J Qwen release profile.",
            recommended_use="Audit and evidence entry for the same paper-consistent deployment line.",
            regression_file="outputs/stage_j/paper_consistent/completion_summary.json",
        ),
    ]
```

Update the `export_stage_k_release(...)` signature and catalog block in `src/stage_k_release.py` to:

```python
def export_stage_k_release(
    export_dir: str | Path,
    *,
    profiles: list[StageKProfile] | None = None,
    materialize: bool = False,
    recommended_profile: str = "default",
    reference_profile: str = "reference",
    title: str = "Stage-K Paper-Consistent Qwen Release",
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    profiles_dir = export_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    active_profiles = profiles or default_stage_k_profiles()
    profile_summaries: list[dict[str, Any]] = []
    for profile in active_profiles:
        source_dir = Path(profile.source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Stage-K source profile does not exist: {source_dir}")
        target_dir = profiles_dir / profile.name
        _ensure_link_or_copy(source_dir, target_dir, materialize=materialize)
        profile_summaries.append(_profile_summary(profile, source_dir))

    catalog = {
        "format": "stage_k_release_v1",
        "stage_lineage": "paper_consistent_stage_j",
        "materialized": bool(materialize),
        "profiles_dir": "profiles",
        "profiles": profile_summaries,
        "recommended_profile": recommended_profile,
        "reference_profile": reference_profile,
    }
```

Update the README intro text in `src/stage_k_release.py` to:

```python
    readme = [
        f"# {title}",
        "",
        "This bundle collects the paper-consistent Qwen deployment-line artifacts.",
        "",
        "Profiles:",
    ]
```

Update the parser default in `scripts/infer_stage_k_release.py` to:

```python
    parser.add_argument("--profile", default="default")
```

Update the `export_stage_k_llama_release(...)` call in `src/stage_k_llama_release.py` to:

```python
    return export_stage_k_release(
        export_dir,
        profiles=default_stage_k_llama_profiles(),
        materialize=materialize,
        recommended_profile="tiny_a",
        reference_profile="stable_reference",
        title="Stage-K Llama Release",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_k_release.py tests/test_stage_k_redesign.py tests/test_stage_k_llama_release.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_stage_k_release.py tests/test_stage_k_redesign.py src/stage_k_llama_release.py src/stage_k_release.py scripts/infer_stage_k_release.py
git commit -m "feat: cut over stage k release defaults to paper consistent line"
```

### Task 2: Switch active Stage K security targets and audits to paper-consistent semantics

**Files:**
- Modify: `tests/test_security_qwen_matrix.py`
- Modify: `tests/test_security_qwen_summary.py`
- Modify: `tests/test_security_qwen_redesign_targets.py`
- Modify: `tests/test_stage_hk_audit.py`
- Modify: `tests/test_stage_i_boundary_audit.py`
- Modify: `src/security_qwen/artifacts.py`
- Modify: `src/security_qwen/matrix.py`
- Modify: `src/security_qwen/vma.py`
- Modify: `src/security_qwen/ima.py`
- Modify: `src/security_qwen/isa.py`
- Modify: `src/security_qwen/tfma.py`
- Modify: `src/stage_hk_audit.py`
- Modify: `src/stage_i_vllm.py`
- Modify: `scripts/security_qwen/export_phase0_templates.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_security_qwen_matrix.py` with:

```python
from src.security_qwen import default_phase0_datasets, default_security_targets, security_matrix_payload


def test_security_targets_cover_paper_consistent_stage_k_release() -> None:
    names = [item.name for item in default_security_targets()]
    assert "stage_a_standard" in names
    assert "stage_h_full_obfuscated" in names
    assert "stage_j_stable_reference" in names
    assert "stage_k_default" in names
    assert "stage_k_reference" in names
    assert "stage_k_tiny_a" not in names


def test_phase0_dataset_catalog_has_three_dataset_kinds() -> None:
    datasets = default_phase0_datasets()
    kinds = [item.kind for item in datasets]
    assert kinds == ["smoke", "inversion", "frequency"]


def test_security_matrix_includes_deployment_intermediate_attacks() -> None:
    payload = security_matrix_payload()
    assert payload["format"] == "qwen_security_matrix_v1"
    deployment_matrix = payload["matrices"]["deployment_intermediate_attacks"]
    assert deployment_matrix["attacks"] == ["isa"]
    assert "kv_cache" in deployment_matrix["observable_types"]
    assert "stage_k_default" in deployment_matrix["targets"]
    assert "stage_k_reference" in deployment_matrix["targets"]
```

Replace `tests/test_security_qwen_summary.py` with:

```python
from pathlib import Path
import json

from src.security_qwen import (
    build_tfma_template,
    build_vma_template,
    get_security_target,
    resolve_security_target,
    security_summary_payload,
)


def test_resolve_security_target_maps_stage_k_default_profile_to_profile_dir() -> None:
    resolved = resolve_security_target("stage_k_default")
    assert resolved.profile == "default"
    assert resolved.resolved_root_dir.endswith("artifacts/stage_k_release/profiles/default")
    assert resolved.server_dir is not None and resolved.server_dir.endswith("/server")
    assert resolved.client_secret_path is not None and resolved.client_secret_path.endswith("client_secret.pt")


def test_resolve_security_target_maps_gate5_scan_artifact() -> None:
    resolved = resolve_security_target("stage_j_tiny_b_scan")
    assert resolved.profile == "tiny_b"
    assert resolved.server_dir is not None and resolved.server_dir.endswith("artifacts/stage_j_gate5_tiny_b/server")
    assert resolved.client_secret_path is not None and resolved.client_secret_path.endswith("artifacts/stage_j_gate5_tiny_b/client/client_secret.pt")


def test_resolve_security_target_maps_gate6_artifact() -> None:
    resolved = resolve_security_target("gate6_targeted_mild")
    assert resolved.profile == "targeted_mild"
    assert resolved.server_dir is not None and resolved.server_dir.endswith("artifacts/stage_j_gate6_targeted_mild/server")
    assert resolved.client_secret_path is not None and resolved.client_secret_path.endswith("artifacts/stage_j_gate6_targeted_mild/client/client_secret.pt")


def test_security_summary_collects_template_results(tmp_path: Path) -> None:
    vma_target = get_security_target("stage_j_stable_reference").to_target()
    tfma_target = get_security_target("stage_k_default").to_target()
    vma_path = tmp_path / "vma" / "stage_j_stable_reference.template.json"
    tfma_path = tmp_path / "tfma" / "stage_k_default.template.json"
    vma_path.parent.mkdir(parents=True, exist_ok=True)
    tfma_path.parent.mkdir(parents=True, exist_ok=True)
    vma_path.write_text(json.dumps(build_vma_template(vma_target), ensure_ascii=False, indent=2), encoding="utf-8")
    tfma_path.write_text(
        json.dumps(build_tfma_template(tfma_target, knowledge_setting="zero_knowledge"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = security_summary_payload(tmp_path)
    assert payload["format"] == "qwen_security_summary_v1"
    assert payload["valid_result_count"] == 2
    assert payload["invalid_result_count"] == 0
    assert payload["by_attack"]["vma"] == 1
    assert payload["by_attack"]["tfma"] == 1


def test_qwen_security_total_report_mentions_legacy_conservative_deployment_line() -> None:
    text = Path("docs/qwen_security/Qwen安全总报告.md").read_text(encoding="utf-8")
    assert "legacy conservative deployment line" in text
```

Replace `tests/test_security_qwen_redesign_targets.py` with:

```python
from src.security_qwen.artifacts import get_security_target, resolve_security_target


def test_redesign_stage_j_target_resolves() -> None:
    spec = get_security_target("stage_j_redesign")
    assert spec.artifact_dir == "artifacts/stage_j_qwen_redesign"


def test_stage_k_reference_target_resolves() -> None:
    resolved = resolve_security_target("stage_k_reference")
    assert resolved.resolved_root_dir.endswith("artifacts/stage_k_release/profiles/reference")


def test_redesign_stage_j_standard_bridge_target_can_be_added_later() -> None:
    from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof

    payload = build_stage_j_standard_weight_proof("artifacts/stage_j_qwen_redesign_standard/server")
    assert payload["is_standard_weight_export"] is True
```

Replace `tests/test_stage_hk_audit.py` with:

```python
from src.stage_hk_audit import build_redesigned_expression_audit


def test_redesigned_expression_audit_detects_stage_h_bootstrap_expression() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_h_source"]["attention_profile_present"] is True
    assert payload["stage_h_source"]["keymat_parameters_present"] is True


def test_redesigned_expression_audit_marks_stage_j_as_paper_consistent_standard_export() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_j"]["candidate_dir"] == "artifacts/stage_j_qwen_paper_consistent"
    assert payload["stage_j"]["has_component_level_expression_manifest"] is True
    assert payload["stage_j"]["has_standard_weight_key_layout"] is True
    assert payload["stage_j"]["completion_status"] == "export_visible_complete"


def test_redesigned_expression_audit_marks_stage_k_as_paper_consistent_packaging() -> None:
    payload = build_redesigned_expression_audit()
    assert payload["stage_k"]["points_to_paper_consistent_stage_j"] is True
```

Replace `tests/test_stage_i_boundary_audit.py` with:

```python
from src.stage_i_vllm import build_stage_i_boundary_audit


def test_stage_i_boundary_audit_tracks_standard_runtime_contract() -> None:
    payload = build_stage_i_boundary_audit()
    assert payload["runtime_graph_is_standard"] is True
    assert payload["custom_online_operator_required"] is False


def test_stage_i_boundary_audit_marks_paper_consistent_expression_proof() -> None:
    payload = build_stage_i_boundary_audit()
    assert payload["exported_artifact_proves_attention_expression"] is True
    assert payload["exported_artifact_proves_ffn_expression"] is True
    assert payload["exported_artifact_proves_norm_expression"] is True
    assert payload["release_catalog_carries_paper_consistent_lineage"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q \
  tests/test_security_qwen_matrix.py \
  tests/test_security_qwen_summary.py \
  tests/test_security_qwen_redesign_targets.py \
  tests/test_stage_hk_audit.py \
  tests/test_stage_i_boundary_audit.py
```

Expected: FAIL because the active `Stage K` security names still use `stable_reference` / `tiny_a`, the release audit still checks redesign lineage, and the boundary audit still reports pre-proof semantics.

- [ ] **Step 3: Write the minimal implementation**

In `src/security_qwen/artifacts.py`, replace the active `Stage K` specs with:

```python
        SecurityTargetSpec(
            name="stage_k_reference",
            stage="K",
            artifact_dir="artifacts/stage_k_release",
            profile="reference",
            variant="paper_consistent_release_profile",
            track="release_paper_consistent",
            supported_attacks=["vma", "ia", "ima", "isa", "tfma", "sda"],
        ),
        SecurityTargetSpec(
            name="stage_k_default",
            stage="K",
            artifact_dir="artifacts/stage_k_release",
            profile="default",
            variant="paper_consistent_release_profile",
            track="release_paper_consistent",
            supported_attacks=["vma", "ia", "ima", "isa", "tfma", "sda"],
        ),
```

In `src/security_qwen/matrix.py`, replace every active `Stage K` target use with:

```python
                    targets["stage_k_reference"].name,
                    targets["stage_k_default"].name,
```

In `src/security_qwen/vma.py`, `src/security_qwen/ima.py`, `src/security_qwen/isa.py`, and `src/security_qwen/tfma.py`, update the default target lists to:

```python
    return [
        "stage_a_standard",
        "stage_h_full_obfuscated",
        "stage_j_stable_reference",
        "stage_j_tiny_a",
        "stage_k_reference",
        "stage_k_default",
    ]
```

In `scripts/security_qwen/export_phase0_templates.py`, replace the `Stage K` template cases with:

```python
        (
            "tfma",
            "stage_k_reference",
            build_tfma_template(
                get_security_target("stage_k_reference").to_target(),
                knowledge_setting="zero_knowledge",
            ),
        ),
        (
            "sda",
            "stage_k_default",
            build_sda_template(
                get_security_target("stage_k_default").to_target(),
                knowledge_setting="domain_aware",
            ),
        ),
```

In `src/stage_hk_audit.py`, replace the `stage_j` / `stage_k` blocks with:

```python
    stage_j_manifest = _load_json("artifacts/stage_j_qwen_paper_consistent/manifest.json")
    stage_j_completion = _load_json("outputs/stage_j/paper_consistent/completion_summary.json")
    stage_k_catalog = _load_json("artifacts/stage_k_release/catalog.json")

    profiles = [item for item in stage_k_catalog.get("profiles", []) if isinstance(item, dict)]
    profile_sources = [item.get("source_dir") for item in profiles if isinstance(item.get("source_dir"), str)]

    stage_j = {
        "candidate_dir": "artifacts/stage_j_qwen_paper_consistent",
        "bootstraps_from_stage_h_pretrained": stage_j_manifest.get("buffered_source_of_truth") == "artifacts/stage_j_qwen_redesign",
        "has_component_level_expression_manifest": "export_visible_components" in stage_j_manifest,
        "has_standard_weight_key_layout": bool(stage_j_manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")),
        "completion_status": stage_j_completion.get("completion_status"),
        "server_dir_present": Path("artifacts/stage_j_qwen_paper_consistent/server").exists(),
        "client_dir_present": Path("artifacts/stage_j_qwen_paper_consistent/client").exists(),
    }

    stage_k = {
        "points_to_paper_consistent_stage_j": stage_k_catalog.get("stage_lineage") == "paper_consistent_stage_j"
        and all(source == "artifacts/stage_j_qwen_paper_consistent" for source in profile_sources),
        "profile_sources": profile_sources,
        "has_expression_metadata_in_catalog": any("manifest" in item for item in profiles),
    }
```

Update the `verdict` / `summary` block in `src/stage_hk_audit.py` to:

```python
    verdict = {
        "expression_enters_bootstrap_source": stage_h_source["attention_profile_present"] and stage_h_source["keymat_parameters_present"],
        "expression_manifest_exists_in_stage_j_export": stage_j["has_component_level_expression_manifest"],
        "expression_is_proven_in_stage_j_export": stage_j["has_standard_weight_key_layout"]
        and stage_j["completion_status"] == "export_visible_complete",
        "expression_is_carried_into_stage_k_release": stage_k["has_expression_metadata_in_catalog"],
    }

    summary = {
        "status": "paper_consistent_release_ready"
        if verdict["expression_is_proven_in_stage_j_export"] and stage_k["points_to_paper_consistent_stage_j"]
        else "expression_audit_inconclusive",
        "next_action": "re-run correctness and security evaluations on the paper-consistent release surface",
    }
```

In `src/stage_i_vllm.py`, replace the return block in `build_stage_i_boundary_audit()` with:

```python
    return {
        "runtime_graph_is_standard": True,
        "custom_online_operator_required": False,
        "compatible_target_surfaces": ["transformers", "vllm", "sglang"],
        "exported_artifact_has_attention_manifest": bool(expression_audit["stage_j"]["has_component_level_expression_manifest"]),
        "exported_artifact_proves_attention_expression": bool(expression_audit["stage_j"]["completion_status"] == "export_visible_complete"),
        "exported_artifact_proves_ffn_expression": bool(expression_audit["stage_j"]["completion_status"] == "export_visible_complete"),
        "exported_artifact_proves_norm_expression": bool(expression_audit["stage_j"]["completion_status"] == "export_visible_complete"),
        "release_catalog_carries_paper_consistent_lineage": bool(
            expression_audit["stage_k"]["points_to_paper_consistent_stage_j"]
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q \
  tests/test_security_qwen_matrix.py \
  tests/test_security_qwen_summary.py \
  tests/test_security_qwen_redesign_targets.py \
  tests/test_stage_hk_audit.py \
  tests/test_stage_i_boundary_audit.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tests/test_security_qwen_matrix.py \
  tests/test_security_qwen_summary.py \
  tests/test_security_qwen_redesign_targets.py \
  tests/test_stage_hk_audit.py \
  tests/test_stage_i_boundary_audit.py \
  src/security_qwen/artifacts.py \
  src/security_qwen/matrix.py \
  src/security_qwen/vma.py \
  src/security_qwen/ima.py \
  src/security_qwen/isa.py \
  src/security_qwen/tfma.py \
  src/stage_hk_audit.py \
  src/stage_i_vllm.py \
  scripts/security_qwen/export_phase0_templates.py
git commit -m "feat: switch stage k security targets to paper consistent release"
```

### Task 3: Recenter Stage K docs and doc tests on the paper-consistent release surface

**Files:**
- Modify: `tests/test_qwen_paper_consistent_docs.py`
- Modify: `docs/阶段K_Qwen交付包装报告.md`
- Modify: `docs/论文一致最终部署主线.md`
- Modify: `README.md`

- [ ] **Step 1: Write the failing doc tests**

Append to `tests/test_qwen_paper_consistent_docs.py`:

```python
def test_stage_k_docs_use_paper_consistent_release_surface() -> None:
    stage_k_text = Path("docs/阶段K_Qwen交付包装报告.md").read_text(encoding="utf-8")
    main_text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    readme_text = Path("README.md").read_text(encoding="utf-8")
    assert "artifacts/stage_k_release" in stage_k_text
    assert "artifacts/stage_j_qwen_paper_consistent" in stage_k_text
    assert "`default`" in stage_k_text
    assert "`reference`" in stage_k_text
    assert "artifacts/stage_k_release" in main_text
    assert "artifacts/stage_k_release" in readme_text


def test_mainline_doc_no_longer_lists_stage_k_cutover_as_remaining_work() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 6\\. 下一步顺序(.*?)## 7\\.", text, re.S)
    assert section is not None
    next_steps = section.group(1)
    assert "唯一 release 面" in next_steps
    assert "correctness / `VMA / IMA / ISA`" in next_steps
    assert "将 `Stage K` 切换到最终论文一致线" not in next_steps
```

Change the `Stage K`-related assertions in `test_readme_uses_single_qwen_deployment_entry()` to:

```python
    assert "artifacts/stage_k_release/" in text
```

- [ ] **Step 2: Run doc tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_qwen_paper_consistent_docs.py
```

Expected: FAIL because the current Stage K docs and README still describe redesign-line release packaging and do not yet expose the paper-consistent release surface.

- [ ] **Step 3: Write the minimal doc updates**

Update `docs/阶段K_Qwen交付包装报告.md` so its core content becomes:

```md
# 阶段 K：Qwen 交付包装报告

> Canonical note: 本文档只回答当前 `Stage K` 的状态，不承担全局主线说明。Qwen 唯一主线入口见 [docs/论文一致最终部署主线.md](论文一致最终部署主线.md)。

## 1. 当前唯一发布面

当前 `Stage K` 只保留一个活跃发布目录：

- `artifacts/stage_k_release`

它包装的唯一源工件是：

- `artifacts/stage_j_qwen_paper_consistent`

## 2. 当前 profile 语义

当前 `Stage K` 使用两个 profile 名称：

- `default`
- `reference`

两者当前都指向同一个 `paper_consistent` `Stage J` 工件，但承担不同入口语义：

- `default`：默认交付入口
- `reference`：审计与证据入口

## 3. 交付职责

当前 `Stage K` 的职责是：

- 提供 release catalog
- 固定 client/server contract
- 提供统一推理入口
- 把 `Stage J` 的论文一致候选工件收口成唯一发布面

## 4. 与历史 Stage K 的关系

旧版 `Stage K` 命名与旧 profile 语义仅作为历史证据保留，不再代表当前唯一论文部署线发布面。
```

Update the `Stage K` and remaining-work sections in `docs/论文一致最终部署主线.md` to:

```md
### Stage K

把最终确认后的 `Stage J` 产物整理成唯一 release 面。

当前唯一 release 面与 profile 入口为：

- `artifacts/stage_k_release`
- `default`
- `reference`
```

```md
## 4. 当前仍未完成的关键项

- 最终论文一致线上的 correctness 仍需在 release 面同口径复跑
- 最终论文一致线上的 `VMA / IMA / ISA` 仍需在 release 面同口径复跑
- `default` / `reference` 当前仍是同源语义别名，后续如有需要可再分化
```

```md
## 6. 下一步顺序

1. 在唯一 release 面上同口径复跑 correctness
2. 在唯一 release 面上复跑 `VMA / IMA / ISA`
3. 根据复跑结果决定是否需要细化 `default` / `reference` 的独立工作点语义
```

Update the Qwen-facing lines in `README.md` to:

```md
- **Qwen：唯一主线为论文一致最终部署线（`Stage J` 与 `Stage K` 已切到 paper-consistent 交付面，最终评测复跑仍在推进）**
```

```md
- 当前 Stage J 唯一候选部署物目录：`artifacts/stage_j_qwen_paper_consistent/`
- 当前 Stage K 唯一发布目录：`artifacts/stage_k_release/`
```

```md
- 当前 Stage K 唯一发布目录：`artifacts/stage_k_release/`
- 当前 Stage K 默认 profile：`default`
```

- [ ] **Step 4: Run doc tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_qwen_paper_consistent_docs.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_qwen_paper_consistent_docs.py docs/阶段K_Qwen交付包装报告.md docs/论文一致最终部署主线.md README.md
git commit -m "docs: recenter stage k on paper consistent release surface"
```

### Task 4: Regenerate the active Stage K release artifact and verify the cutover end to end

**Files:**
- Modify: `artifacts/stage_k_release/catalog.json`
- Modify: `artifacts/stage_k_release/README.md`
- Modify: `artifacts/stage_k_release/deployment_contract.json`
- Modify: `artifacts/stage_k_release/profiles/default`
- Modify: `artifacts/stage_k_release/profiles/reference`

- [ ] **Step 1: Regenerate the active Stage K release artifact**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_k_release.py
```

Expected:

- `artifacts/stage_k_release/catalog.json` contains `stage_lineage = "paper_consistent_stage_j"`
- `profiles/default`
- `profiles/reference`

- [ ] **Step 2: Inspect the generated catalog**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

catalog = json.loads(Path("artifacts/stage_k_release/catalog.json").read_text(encoding="utf-8"))
print(catalog["stage_lineage"])
print(catalog["recommended_profile"])
print(catalog["reference_profile"])
print([item["name"] for item in catalog["profiles"]])
print(sorted({item["source_dir"] for item in catalog["profiles"]}))
PY
```

Expected output:

```text
paper_consistent_stage_j
default
reference
['default', 'reference']
['artifacts/stage_j_qwen_paper_consistent']
```

- [ ] **Step 3: Run the focused verification suite**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q \
  tests/test_stage_k_release.py \
  tests/test_stage_k_redesign.py \
  tests/test_stage_k_llama_release.py \
  tests/test_security_qwen_matrix.py \
  tests/test_security_qwen_summary.py \
  tests/test_security_qwen_redesign_targets.py \
  tests/test_stage_hk_audit.py \
  tests/test_stage_i_boundary_audit.py \
  tests/test_qwen_paper_consistent_docs.py
git diff --check
```

Expected:

- pytest PASS
- `git diff --check` prints nothing

- [ ] **Step 4: Commit the regenerated artifact and remaining cutover changes**

```bash
git add artifacts/stage_k_release
git commit -m "chore: regenerate stage k paper consistent release"
```
