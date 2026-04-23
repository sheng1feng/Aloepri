# Qwen Stage J Export-Visible Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export one canonical `paper_consistent` Stage-J candidate directly from the redesign source, generate a unified export-visible evidence package for it, and update canonical docs/tests so this candidate becomes the only active Stage-J acceptance target.

**Architecture:** Reuse the existing Stage-J standardization machinery to materialize a new `artifacts/stage_j_qwen_paper_consistent/` artifact from `artifacts/stage_j_qwen_redesign/`, but do not reuse the historical bridge semantics as the acceptance target. Build a new unified proof bundle under `outputs/stage_j/paper_consistent/` that aggregates standard-weight proof, attention/FFN/norm export-visible proofs, correctness regression, and a single completion summary.

**Tech Stack:** Python, PyTorch, Transformers, safetensors, existing `src/stage_j_standard_bridge.py`, existing `src/stage_j_standard_weight_proof.py`, pytest.

---

### Task 1: Upgrade the Stage-J paper-consistent descriptor to the canonical candidate model

**Files:**
- Modify: `src/stage_j_paper_consistent.py`
- Modify: `tests/test_stage_j_paper_consistent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage_j_paper_consistent.py`:

```python
from src.stage_j_paper_consistent import build_stage_j_paper_consistent_target


def test_stage_j_paper_consistent_target_uses_canonical_candidate_paths() -> None:
    payload = build_stage_j_paper_consistent_target()
    assert payload["canonical_candidate_dir"] == "artifacts/stage_j_qwen_paper_consistent"
    assert payload["evidence_dir"] == "outputs/stage_j/paper_consistent"
    assert payload["historical_bridge_dir"] == "artifacts/stage_j_qwen_redesign_standard"
    assert payload["completion_statuses"] == ["export_visible_complete", "not_complete"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

Expected: FAIL because the current target descriptor only defines the old legacy fields.

- [ ] **Step 3: Write minimal implementation**

Update `src/stage_j_paper_consistent.py` so the descriptor returns the new canonical candidate metadata:

```python
from __future__ import annotations

from typing import Any


def build_stage_j_paper_consistent_target() -> dict[str, Any]:
    return {
        "stage": "J",
        "goal": "paper_consistent_standard_deployable_obfuscated_checkpoint",
        "standard_graph_required": True,
        "standard_visible_keys_required": True,
        "bridge_is_final_target": False,
        "buffered_reference": "artifacts/stage_j_qwen_redesign",
        "historical_bridge_dir": "artifacts/stage_j_qwen_redesign_standard",
        "canonical_candidate_dir": "artifacts/stage_j_qwen_paper_consistent",
        "evidence_dir": "outputs/stage_j/paper_consistent",
        "required_evidence_files": [
            "standard_weight_proof.json",
            "attention_export_visible_proof.json",
            "ffn_export_visible_proof.json",
            "norm_export_visible_proof.json",
            "correctness_regression.json",
            "completion_summary.json",
        ],
        "completion_statuses": ["export_visible_complete", "not_complete"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stage_j_paper_consistent.py tests/test_stage_j_paper_consistent.py
git commit -m "feat: define canonical stage j paper consistent candidate"
```

### Task 2: Export the canonical `stage_j_qwen_paper_consistent` candidate

**Files:**
- Modify: `src/stage_j_paper_consistent.py`
- Create: `scripts/export_stage_j_paper_consistent_checkpoint.py`
- Modify: `tests/test_stage_j_paper_consistent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage_j_paper_consistent.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.stage_j_paper_consistent import export_stage_j_paper_consistent_candidate


def test_export_stage_j_paper_consistent_candidate_writes_canonical_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_j_qwen_redesign"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "server" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (source_dir / "server" / "obfuscation_config.json").write_text(
        json.dumps(
            {
                "attention_profile": "rqk_hqk_block_taukv_taugroup",
                "lambda": 0.3,
                "h": 128,
                "alpha_e": 0.1,
                "alpha_h": 0.05,
                "adapted_layers": [0, 1],
                "beta": 0.25,
                "gamma": 0.75,
                "kappa_overrides": {"layers": {"0": {"input": 2.0, "post_attn": 3.0}}, "final": 4.0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "buffer::stage_a_model.model.embed_tokens.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.lm_head.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.input_layernorm.metric_matrix": torch.diag(torch.tensor([4.0] * 12)),
            "buffer::stage_a_model.model.layers.0.post_attention_layernorm.metric_matrix": torch.diag(torch.tensor([9.0] * 12)),
            "buffer::stage_a_model.model.norm.metric_matrix": torch.diag(torch.tensor([16.0] * 12)),
            "buffer::stage_a_model.model.layers.0.self_attn.q_weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.k_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.v_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.o_weight": torch.zeros(12, 8),
            "buffer::stage_a_model.model.layers.0.mlp.gate_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.up_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.down_weight": torch.zeros(12, 16),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_paper_consistent"
    result = export_stage_j_paper_consistent_candidate(export_dir, source_dir=source_dir)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["export_dir"] == export_dir
    assert manifest["track"] == "paper_consistent_candidate"
    assert manifest["standard_weight_proof"]["is_standard_weight_export"] is True
    assert manifest["bridge_is_acceptance_target"] is False
    assert manifest["candidate_role"] == "canonical_stage_j_acceptance_target"


def test_export_stage_j_paper_consistent_candidate_carries_component_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "stage_j_qwen_redesign"
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (source_dir / "server" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 896,
                "intermediate_size": 4864,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "vocab_size": 8,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (source_dir / "server" / "obfuscation_config.json").write_text(
        json.dumps(
            {
                "attention_profile": "rqk_hqk_block_taukv_taugroup",
                "lambda": 0.3,
                "h": 128,
                "adapted_layers": [0, 1, 2],
                "beta": 0.25,
                "gamma": 0.75,
                "kappa_overrides": {"layers": {"0": {"input": 2.0, "post_attn": 3.0}}, "final": 4.0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "buffer::stage_a_model.model.embed_tokens.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.lm_head.weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.input_layernorm.metric_matrix": torch.diag(torch.tensor([4.0] * 12)),
            "buffer::stage_a_model.model.layers.0.post_attention_layernorm.metric_matrix": torch.diag(torch.tensor([9.0] * 12)),
            "buffer::stage_a_model.model.norm.metric_matrix": torch.diag(torch.tensor([16.0] * 12)),
            "buffer::stage_a_model.model.layers.0.self_attn.q_weight": torch.zeros(8, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.k_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.v_weight": torch.zeros(4, 12),
            "buffer::stage_a_model.model.layers.0.self_attn.o_weight": torch.zeros(12, 8),
            "buffer::stage_a_model.model.layers.0.mlp.gate_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.up_weight": torch.zeros(16, 12),
            "buffer::stage_a_model.model.layers.0.mlp.down_weight": torch.zeros(12, 16),
        },
        str(source_dir / "server" / "model.safetensors"),
    )
    (source_dir / "client" / "client_secret.pt").write_text("secret", encoding="utf-8")

    export_dir = tmp_path / "stage_j_qwen_paper_consistent"
    export_stage_j_paper_consistent_candidate(export_dir, source_dir=source_dir)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_visible_components"]["attention"]["profile"] == "rqk_hqk_block_taukv_taugroup"
    assert manifest["export_visible_components"]["ffn"]["adapted_layers_count"] == 3
    assert manifest["export_visible_components"]["norm"]["strategy"] == "kappa_fused"
    assert manifest["export_visible_components"]["norm"]["has_kappa_overrides"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

Expected: FAIL because no canonical candidate exporter exists yet.

- [ ] **Step 3: Write minimal implementation**

Extend `src/stage_j_paper_consistent.py` with a canonical exporter that reuses the existing standardization machinery but rewrites the manifest semantics:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_export_visible_component_metadata(source_server: Path, norm_strategy: str) -> dict[str, Any]:
    obf_cfg = _load_json(source_server / "obfuscation_config.json")
    return {
        "attention": {
            "profile": obf_cfg.get("attention_profile"),
            "has_profile": bool(obf_cfg.get("attention_profile")),
            "has_head_group_semantics": "taugroup" in str(obf_cfg.get("attention_profile", "")),
            "has_block_semantics": "block" in str(obf_cfg.get("attention_profile", "")),
        },
        "ffn": {
            "adapted_layers_count": len(obf_cfg.get("adapted_layers", [])),
            "beta": obf_cfg.get("beta"),
            "gamma": obf_cfg.get("gamma"),
        },
        "norm": {
            "strategy": norm_strategy,
            "has_kappa_overrides": "kappa_overrides" in obf_cfg,
        },
    }


def export_stage_j_paper_consistent_candidate(
    export_dir: str | Path,
    *,
    source_dir: str | Path = "artifacts/stage_j_qwen_redesign",
    materialize: bool = False,
    norm_strategy: str = "kappa_fused",
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    source_dir = Path(source_dir)
    bridge_result = export_stage_j_redesign_standard_bridge(
        export_dir,
        source_dir=source_dir,
        materialize=materialize,
        norm_strategy=norm_strategy,
    )
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["track"] = "paper_consistent_candidate"
    manifest["goal"] = "paper_consistent_standard_deployable_obfuscated_checkpoint"
    manifest["candidate_role"] = "canonical_stage_j_acceptance_target"
    manifest["bridge_is_acceptance_target"] = False
    manifest["historical_bridge_reference"] = "artifacts/stage_j_qwen_redesign_standard"
    manifest["buffered_source_of_truth"] = "artifacts/stage_j_qwen_redesign"
    manifest["export_visible_components"] = _build_export_visible_component_metadata(
        Path(source_dir) / "server",
        manifest["norm_strategy"],
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return bridge_result
```

Create `scripts/export_stage_j_paper_consistent_checkpoint.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_paper_consistent import export_stage_j_paper_consistent_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the canonical Stage-J paper-consistent candidate.")
    parser.add_argument("--export-dir", default="artifacts/stage_j_qwen_paper_consistent")
    parser.add_argument("--source-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--norm-strategy", default="kappa_fused", choices=["ones", "metric_diag_sqrt", "kappa_fused"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_stage_j_paper_consistent_candidate(
        args.export_dir,
        source_dir=args.source_dir,
        materialize=args.materialize,
        norm_strategy=args.norm_strategy,
    )
    print(f"Exported Stage-J paper-consistent candidate to {result['export_dir']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stage_j_paper_consistent.py scripts/export_stage_j_paper_consistent_checkpoint.py tests/test_stage_j_paper_consistent.py
git commit -m "feat: export canonical stage j paper consistent candidate"
```

### Task 3: Build the unified Stage-J export-visible evidence package

**Files:**
- Modify: `src/stage_j_paper_consistent.py`
- Create: `scripts/run_stage_j_paper_consistent_completion.py`
- Modify: `tests/test_stage_j_paper_consistent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage_j_paper_consistent.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from src.stage_j_paper_consistent import (
    build_stage_j_paper_consistent_completion_summary,
    build_stage_j_paper_consistent_evidence_bundle,
)


def test_build_stage_j_paper_consistent_evidence_bundle_writes_all_required_reports(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "artifacts" / "stage_j_qwen_paper_consistent"
    source_dir = tmp_path / "artifacts" / "stage_j_qwen_redesign"
    (candidate_dir / "server").mkdir(parents=True)
    (candidate_dir / "client").mkdir(parents=True)
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "track": "paper_consistent_candidate",
                "norm_strategy": "kappa_fused",
                "standard_weight_proof": {"is_standard_weight_export": True, "layout": "standard_weight_visible"},
                "export_visible_components": {
                    "attention": {"profile": "rqk_hqk_block_taukv_taugroup", "has_profile": True, "has_head_group_semantics": True, "has_block_semantics": True},
                    "ffn": {"adapted_layers_count": 2, "beta": 0.25, "gamma": 0.75},
                    "norm": {"strategy": "kappa_fused", "has_kappa_overrides": True},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=candidate_dir,
        source_dir=source_dir,
        output_dir=tmp_path / "outputs" / "stage_j" / "paper_consistent",
        correctness_payload={
            "summary": {
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
                "avg_restored_full_logits_max_abs_error": 0.0,
            }
        },
    )

    assert bundle["standard_weight_proof"]["status"] == "pass"
    assert bundle["attention_export_visible_proof"]["status"] == "pass"
    assert bundle["ffn_export_visible_proof"]["status"] == "pass"
    assert bundle["norm_export_visible_proof"]["status"] == "pass"
    assert bundle["correctness_regression"]["status"] == "pass"


def test_build_stage_j_paper_consistent_completion_summary_returns_export_visible_complete() -> None:
    summary = build_stage_j_paper_consistent_completion_summary(
        {
            "standard_weight_proof": {"status": "pass"},
            "attention_export_visible_proof": {"status": "pass"},
            "ffn_export_visible_proof": {"status": "pass"},
            "norm_export_visible_proof": {"status": "pass"},
            "correctness_regression": {"status": "pass"},
        }
    )
    assert summary["completion_status"] == "export_visible_complete"
    assert summary["blocking_components"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

Expected: FAIL because the unified bundle and completion summary builders do not exist.

- [ ] **Step 3: Write minimal implementation**

Extend `src/stage_j_paper_consistent.py` with evidence builders:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluator import write_json
from src.stage_j_bridge_regression import run_stage_j_bridge_regression
from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof


def build_stage_j_attention_export_visible_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    attention = manifest.get("export_visible_components", {}).get("attention", {})
    passed = bool(
        manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")
        and attention.get("has_profile")
        and attention.get("has_head_group_semantics")
        and attention.get("has_block_semantics")
    )
    return {
        "status": "pass" if passed else "fail",
        "profile": attention.get("profile"),
        "has_head_group_semantics": bool(attention.get("has_head_group_semantics")),
        "has_block_semantics": bool(attention.get("has_block_semantics")),
    }


def build_stage_j_ffn_export_visible_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    ffn = manifest.get("export_visible_components", {}).get("ffn", {})
    passed = bool(
        manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")
        and int(ffn.get("adapted_layers_count", 0)) > 0
    )
    return {
        "status": "pass" if passed else "fail",
        "adapted_layers_count": int(ffn.get("adapted_layers_count", 0)),
        "beta": ffn.get("beta"),
        "gamma": ffn.get("gamma"),
    }


def build_stage_j_norm_export_visible_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    norm = manifest.get("export_visible_components", {}).get("norm", {})
    strategy = norm.get("strategy")
    passed = bool(
        manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")
        and strategy in {"kappa_fused", "metric_diag_sqrt"}
        and (bool(norm.get("has_kappa_overrides")) or strategy == "metric_diag_sqrt")
    )
    return {
        "status": "pass" if passed else "fail",
        "strategy": strategy,
        "has_kappa_overrides": bool(norm.get("has_kappa_overrides")),
    }


def build_stage_j_correctness_regression_status(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    passed = bool(
        float(summary.get("generated_ids_exact_match_rate", 0.0)) > 0.0
        and float(summary.get("generated_text_exact_match_rate", 0.0)) > 0.0
    )
    return {"status": "pass" if passed else "fail", **summary}


def build_stage_j_paper_consistent_completion_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    blocking = [name for name, payload in bundle.items() if payload.get("status") != "pass"]
    return {
        "completion_status": "export_visible_complete" if not blocking else "not_complete",
        "blocking_components": blocking,
    }


def build_stage_j_paper_consistent_evidence_bundle(
    *,
    candidate_dir: str | Path = "artifacts/stage_j_qwen_paper_consistent",
    source_dir: str | Path = "artifacts/stage_j_qwen_redesign",
    output_dir: str | Path = "outputs/stage_j/paper_consistent",
    correctness_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["standard_weight_proof"] = build_stage_j_standard_weight_proof(candidate_dir / "server")

    if correctness_payload is None:
        correctness_payload = run_stage_j_bridge_regression(
            buffered_server_dir=str((source_dir / "server").resolve()),
            bridge_server_dir=str((candidate_dir / "server").resolve()),
        )

    bundle = {
        "standard_weight_proof": {
            "status": "pass" if manifest["standard_weight_proof"]["is_standard_weight_export"] else "fail",
            **manifest["standard_weight_proof"],
        },
        "attention_export_visible_proof": build_stage_j_attention_export_visible_proof(manifest),
        "ffn_export_visible_proof": build_stage_j_ffn_export_visible_proof(manifest),
        "norm_export_visible_proof": build_stage_j_norm_export_visible_proof(manifest),
        "correctness_regression": build_stage_j_correctness_regression_status(correctness_payload),
    }
    summary = build_stage_j_paper_consistent_completion_summary(bundle)
    bundle["completion_summary"] = summary

    for name, payload in bundle.items():
        write_json(output_dir / f"{name}.json", payload)
    return bundle
```

Create `scripts/run_stage_j_paper_consistent_completion.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_paper_consistent import build_stage_j_paper_consistent_evidence_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the unified Stage-J paper-consistent completion evidence bundle.")
    parser.add_argument("--candidate-dir", default="artifacts/stage_j_qwen_paper_consistent")
    parser.add_argument("--source-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--output-dir", default="outputs/stage_j/paper_consistent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=args.candidate_dir,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    print(f"Built Stage-J paper-consistent evidence bundle with status {payload['completion_summary']['completion_status']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stage_j_paper_consistent.py scripts/run_stage_j_paper_consistent_completion.py tests/test_stage_j_paper_consistent.py
git commit -m "feat: add stage j paper consistent evidence bundle"
```

### Task 4: Update canonical docs and doc tests to center the new Stage-J candidate

**Files:**
- Modify: `docs/阶段J_论文一致部署路线说明.md`
- Modify: `docs/论文一致最终部署主线.md`
- Modify: `README.md`
- Modify: `tests/test_qwen_paper_consistent_docs.py`

- [ ] **Step 1: Write the failing doc tests**

Append to `tests/test_qwen_paper_consistent_docs.py`:

```python
def test_stage_j_docs_use_paper_consistent_candidate_as_active_target() -> None:
    route_text = Path("docs/阶段J_论文一致部署路线说明.md").read_text(encoding="utf-8")
    main_text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    readme_text = Path("README.md").read_text(encoding="utf-8")
    assert "artifacts/stage_j_qwen_paper_consistent" in route_text
    assert "artifacts/stage_j_qwen_paper_consistent" in readme_text
    assert "outputs/stage_j/paper_consistent/completion_summary.json" in main_text


def test_stage_j_docs_demote_historical_bridge_to_auxiliary_evidence() -> None:
    route_text = Path("docs/阶段J_论文一致部署路线说明.md").read_text(encoding="utf-8")
    assert "历史中间线" in route_text
    assert "artifacts/stage_j_qwen_redesign_standard" in route_text
```

- [ ] **Step 2: Run doc tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_qwen_paper_consistent_docs.py
```

Expected: FAIL because the canonical docs do not yet refer to the new candidate/evidence package.

- [ ] **Step 3: Write minimal doc updates**

Update `docs/阶段J_论文一致部署路线说明.md` so that:

- the active `Stage J` target is `artifacts/stage_j_qwen_paper_consistent`;
- the historical bridge `artifacts/stage_j_qwen_redesign_standard` is described as `历史中间线` or equivalent subordinate wording;
- the route doc names the unified evidence package under `outputs/stage_j/paper_consistent/`.

Update `docs/论文一致最终部署主线.md` so that:

- the `Stage J` section names `artifacts/stage_j_qwen_paper_consistent` as the unique candidate;
- the current evidence section includes `outputs/stage_j/paper_consistent/completion_summary.json`.

Update `README.md` so that:

- the Qwen-facing `Stage J` wording points to the canonical candidate path rather than historical `stage_j_full_square*` or the historical bridge.

- [ ] **Step 4: Run doc tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_qwen_paper_consistent_docs.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/阶段J_论文一致部署路线说明.md docs/论文一致最终部署主线.md README.md tests/test_qwen_paper_consistent_docs.py
git commit -m "docs: center stage j on paper consistent candidate"
```

### Task 5: Verify the end-to-end Stage-J completion lane

**Files:**
- Modify: `tests/test_stage_j_paper_consistent.py`

- [ ] **Step 1: Add one end-to-end lane test**

Append to `tests/test_stage_j_paper_consistent.py`:

```python
def test_stage_j_paper_consistent_bundle_reports_explicit_completion_state(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "artifacts" / "stage_j_qwen_paper_consistent"
    source_dir = tmp_path / "artifacts" / "stage_j_qwen_redesign"
    (candidate_dir / "server").mkdir(parents=True)
    (candidate_dir / "client").mkdir(parents=True)
    (source_dir / "server").mkdir(parents=True)
    (source_dir / "client").mkdir(parents=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "track": "paper_consistent_candidate",
                "standard_weight_proof": {"is_standard_weight_export": True, "layout": "standard_weight_visible"},
                "export_visible_components": {
                    "attention": {"profile": "rqk_hqk_block_taukv_taugroup", "has_profile": True, "has_head_group_semantics": True, "has_block_semantics": True},
                    "ffn": {"adapted_layers_count": 2, "beta": 0.25, "gamma": 0.75},
                    "norm": {"strategy": "kappa_fused", "has_kappa_overrides": True},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = build_stage_j_paper_consistent_evidence_bundle(
        candidate_dir=candidate_dir,
        source_dir=source_dir,
        output_dir=tmp_path / "outputs" / "stage_j" / "paper_consistent",
        correctness_payload={
            "summary": {
                "generated_ids_exact_match_rate": 1.0,
                "generated_text_exact_match_rate": 1.0,
                "avg_restored_full_logits_max_abs_error": 0.0,
            }
        },
    )
    assert bundle["completion_summary"]["completion_status"] == "export_visible_complete"
```

- [ ] **Step 2: Run the focused test suites**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py tests/test_qwen_paper_consistent_docs.py
```

Expected: PASS.

- [ ] **Step 3: Run the end-to-end scripts**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_paper_consistent_checkpoint.py --export-dir artifacts/stage_j_qwen_paper_consistent --source-dir artifacts/stage_j_qwen_redesign
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_j_paper_consistent_completion.py --candidate-dir artifacts/stage_j_qwen_paper_consistent --source-dir artifacts/stage_j_qwen_redesign --output-dir outputs/stage_j/paper_consistent
```

Expected:

- the export script writes `artifacts/stage_j_qwen_paper_consistent/`;
- the completion script writes all six evidence files under `outputs/stage_j/paper_consistent/`.

- [ ] **Step 4: Run final verification**

Run:

```bash
conda run --no-capture-output -n qwen-transformers python -m pytest -q tests/test_stage_j_paper_consistent.py tests/test_qwen_paper_consistent_docs.py
git diff --check
```

Expected:

- pytest passes;
- `git diff --check` returns no output.

- [ ] **Step 5: Commit**

```bash
git add tests/test_stage_j_paper_consistent.py
git commit -m "test: verify stage j export visible completion lane"
```
