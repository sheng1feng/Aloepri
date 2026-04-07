from __future__ import annotations

from pathlib import Path
from typing import Any

from src.stage_k_release import export_stage_k_release


def export_release_bundle(
    export_dir: str | Path,
    *,
    materialize: bool = False,
) -> dict[str, Any]:
    return export_stage_k_release(export_dir, materialize=materialize)
