from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageKProfile:
    name: str
    source_dir: str
    description: str
    recommended_use: str
    correctness_evidence_file: str | None = None


def default_stage_k_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="default",
            source_dir="artifacts/stage_j_qwen_paper_consistent",
            description="Default paper-consistent Stage-J Qwen release profile.",
            recommended_use="Default delivery entry for the paper-consistent Qwen deployment line.",
            correctness_evidence_file="outputs/stage_j/paper_consistent/correctness_regression.json",
        ),
        StageKProfile(
            name="reference",
            source_dir="artifacts/stage_j_qwen_paper_consistent",
            description="Reference paper-consistent Stage-J Qwen release profile.",
            recommended_use="Audit and evidence entry for the same paper-consistent deployment line.",
            correctness_evidence_file="outputs/stage_j/paper_consistent/correctness_regression.json",
        ),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_summary(profile: StageKProfile, source_dir: Path) -> dict[str, Any]:
    metadata = _load_json(source_dir / "stage_i_metadata.json")
    manifest = _load_json(source_dir / "manifest.json")
    correctness = {}
    if profile.correctness_evidence_file is not None:
        candidate = Path(profile.correctness_evidence_file)
        if candidate.exists():
            correctness = _load_json(candidate)
    return {
        "name": profile.name,
        "description": profile.description,
        "recommended_use": profile.recommended_use,
        "source_dir": str(source_dir),
        "server_dir": f"profiles/{profile.name}/server",
        "client_secret": f"profiles/{profile.name}/client/client_secret.pt",
        "metadata": metadata,
        "manifest": manifest,
        "correctness_evidence_file": profile.correctness_evidence_file,
        "correctness_summary": correctness,
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
    (export_dir / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    deployment_contract = {
        "client_holds": ["client_secret.pt"],
        "server_holds": ["server/config.json", "server/generation_config.json", "server/model.safetensors", "server/tokenizer.json", "server/tokenizer_config.json", "server/chat_template.jinja"],
        "input_mapping_function": "src/transforms.py::map_input_ids",
        "output_restore_function": "src/transforms.py::restore_logits",
        "token_id_restore_function": "src/transforms.py::unmap_output_ids",
    }
    (export_dir / "deployment_contract.json").write_text(
        json.dumps(deployment_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = [
        f"# {title}",
        "",
        "This bundle collects the paper-consistent Qwen deployment-line artifacts.",
        "",
        "Profiles:",
    ]
    for item in profile_summaries:
        readme.append(f"- `{item['name']}`: {item['description']}")
        readme.append(f"  - Recommended use: {item['recommended_use']}")
        summary = item.get("correctness_summary", {})
        if summary:
            readme.append(
                f"  - Correctness evidence: status={summary.get('status')}, "
                f"gen_ids_match={summary.get('generated_ids_exact_match_rate')}, "
                f"gen_text_match={summary.get('generated_text_exact_match_rate')}"
            )
    readme.append("")
    readme.append("Use `scripts/infer_stage_k_release.py` to run inference by profile name.")
    (export_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return catalog
