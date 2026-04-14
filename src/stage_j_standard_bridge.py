from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof


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


def build_stage_j_standard_bridge_manifest() -> dict[str, Any]:
    return {
        "stage": "J",
        "track": "standard_visible_bridge",
        "goal": "standard_weight_visible_bridge_for_redesigned_stage_j",
        "buffered_reference": "artifacts/stage_j_qwen_redesign",
        "standard_visible_source": "artifacts/stage_j_full_square",
        "equivalence_to_buffered_redesign": False,
        "notes": "Bridge artifact: standard-weight-visible, but not yet proven equivalent to the buffered redesign line.",
    }


def export_stage_j_redesign_standard_bridge(
    export_dir: str | Path,
    *,
    source_dir: str | Path = "artifacts/stage_j_full_square",
    materialize: bool = False,
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Stage-J standard bridge source does not exist: {source_dir}")

    server_source = source_dir / "server"
    client_source = source_dir / "client"
    if not server_source.exists() or not client_source.exists():
        raise FileNotFoundError("Stage-J standard bridge source must contain server/ and client/ directories.")

    export_dir.mkdir(parents=True, exist_ok=True)
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    _ensure_link_or_copy(server_source, server_dir, materialize=materialize)
    _ensure_link_or_copy(client_source, client_dir, materialize=materialize)

    manifest = build_stage_j_standard_bridge_manifest()
    manifest["materialized"] = bool(materialize)
    manifest["resolved_source_dir"] = str(source_dir)
    manifest["server_dir"] = "server"
    manifest["client_dir"] = "client"
    manifest["standard_weight_proof"] = build_stage_j_standard_weight_proof(server_dir)
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "export_dir": export_dir,
        "server_dir": server_dir,
        "client_dir": client_dir,
        "manifest_path": export_dir / "manifest.json",
    }
