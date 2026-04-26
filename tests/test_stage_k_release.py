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
