from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def build_stage_j_redesign_manifest() -> dict[str, Any]:
    return {
        "stage": "J",
        "goal": "qwen_full_model_deployment_materialization",
        "source_stages": ["H", "I"],
        "bootstrap_source": "artifacts/stage_h_pretrained",
        "legacy_reference": "artifacts/stage_j_full_square",
        "redesign_export_dir": "artifacts/stage_j_qwen_redesign",
    }


def _ensure_link_or_copy(source_dir: Path, target_dir: Path, *, materialize: bool) -> None:
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
    if materialize:
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)


def export_stage_j_redesign_checkpoint(
    export_dir: str | Path,
    *,
    source_dir: str | Path = "artifacts/stage_h_pretrained",
    materialize: bool = False,
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Stage-J redesign bootstrap source does not exist: {source_dir}")

    server_source = source_dir / "server"
    client_source = source_dir / "client"
    if not server_source.exists() or not client_source.exists():
        raise FileNotFoundError("Stage-J redesign bootstrap source must contain server/ and client/ directories.")

    export_dir.mkdir(parents=True, exist_ok=True)
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    _ensure_link_or_copy(server_source, server_dir, materialize=materialize)
    _ensure_link_or_copy(client_source, client_dir, materialize=materialize)

    manifest = build_stage_j_redesign_manifest()
    manifest["materialized"] = bool(materialize)
    manifest["resolved_bootstrap_source"] = str(source_dir)
    manifest["server_dir"] = "server"
    manifest["client_dir"] = "client"
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "export_dir": export_dir,
        "server_dir": server_dir,
        "client_dir": client_dir,
        "manifest_path": export_dir / "manifest.json",
    }


def build_stage_j_redesign_regression(export_dir: str | Path = "artifacts/stage_j_qwen_redesign") -> dict[str, Any]:
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "stage": "J",
        "artifact_dir": str(export_dir),
        "checks": {
            "manifest_exists": manifest_path.exists(),
            "server_exists": (export_dir / "server").exists(),
            "client_exists": (export_dir / "client").exists(),
            "source_stages_match": manifest.get("source_stages") == ["H", "I"],
        },
        "summary": {
            "status": "ready" if manifest_path.exists() and (export_dir / "server").exists() and (export_dir / "client").exists() else "missing_artifact",
        },
    }
