from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.security_qwen.schema import SecurityEvalTarget


@dataclass(frozen=True)
class SecurityTargetSpec:
    name: str
    stage: str
    artifact_dir: str
    profile: str | None
    variant: str
    track: str
    supported_attacks: list[str]

    def to_target(self) -> SecurityEvalTarget:
        return SecurityEvalTarget(
            stage=self.stage,
            artifact_dir=self.artifact_dir,
            profile=self.profile,
            model_family="qwen",
            variant=self.variant,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target"] = self.to_target().to_dict()
        return payload


@dataclass(frozen=True)
class SecurityResolvedTarget:
    name: str
    stage: str
    track: str
    artifact_dir: str
    resolved_root_dir: str
    profile: str | None
    variant: str
    server_dir: str | None
    client_secret_path: str | None
    metadata_path: str | None
    manifest_path: str | None
    release_catalog_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_security_targets() -> list[SecurityTargetSpec]:
    all_attacks = ["vma", "ia", "ima", "isa", "tfma", "sda"]
    return [
        SecurityTargetSpec(
            name="stage_a_standard",
            stage="A",
            artifact_dir="artifacts/stage_i_vllm",
            profile=None,
            variant="stage_a_vocab_permutation",
            track="deployment_baseline",
            supported_attacks=["ima"],
        ),
        SecurityTargetSpec(
            name="stage_h_full_obfuscated",
            stage="H",
            artifact_dir="artifacts/stage_h_full_obfuscated",
            profile=None,
            variant="keymat_full_obfuscated",
            track="research",
            supported_attacks=all_attacks,
        ),
        SecurityTargetSpec(
            name="stage_j_stable_reference",
            stage="J",
            artifact_dir="artifacts/stage_j_full_square",
            profile="stable_reference",
            variant="standard_shape_full_layer",
            track="deployment",
            supported_attacks=all_attacks,
        ),
        SecurityTargetSpec(
            name="stage_j_tiny_a",
            stage="J",
            artifact_dir="artifacts/stage_j_full_square_tiny_a",
            profile="tiny_a",
            variant="standard_shape_full_layer",
            track="deployment",
            supported_attacks=all_attacks,
        ),
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
        SecurityTargetSpec(
            name="stage_j_redesign",
            stage="J",
            artifact_dir="artifacts/stage_j_qwen_redesign",
            profile=None,
            variant="redesigned_deployment_materialization",
            track="deployment_redesign",
            supported_attacks=["vma", "ima", "isa"],
        ),
        SecurityTargetSpec(
            name="stage_j_tiny_b_scan",
            stage="J",
            artifact_dir="artifacts/stage_j_gate5_tiny_b",
            profile="tiny_b",
            variant="standard_shape_full_layer",
            track="deployment_scan",
            supported_attacks=["vma", "ima", "isa", "tfma", "sda"],
        ),
        SecurityTargetSpec(
            name="stage_j_small_a_scan",
            stage="J",
            artifact_dir="artifacts/stage_j_gate5_small_a",
            profile="small_a",
            variant="standard_shape_full_layer",
            track="deployment_scan",
            supported_attacks=["vma", "ima", "isa", "tfma", "sda"],
        ),
        SecurityTargetSpec(
            name="stage_j_paper_like_scan",
            stage="J",
            artifact_dir="artifacts/stage_j_gate5_paper_like",
            profile="paper_like",
            variant="standard_shape_full_layer",
            track="deployment_scan",
            supported_attacks=["vma", "ima", "isa", "tfma", "sda"],
        ),
        SecurityTargetSpec(
            name="gate6_targeted_mild",
            stage="J",
            artifact_dir="artifacts/stage_j_gate6_targeted_mild",
            profile="targeted_mild",
            variant="gate6_targeted_sensitive_square_transform",
            track="enhancement",
            supported_attacks=["vma", "ima"],
        ),
        SecurityTargetSpec(
            name="gate6_targeted_strong",
            stage="J",
            artifact_dir="artifacts/stage_j_gate6_targeted_strong",
            profile="targeted_strong",
            variant="gate6_targeted_sensitive_square_transform",
            track="enhancement",
            supported_attacks=["vma", "ima"],
        ),
        SecurityTargetSpec(
            name="gate6_targeted_extreme",
            stage="J",
            artifact_dir="artifacts/stage_j_gate6_targeted_extreme",
            profile="targeted_extreme",
            variant="gate6_targeted_sensitive_square_transform",
            track="enhancement",
            supported_attacks=["vma", "ima"],
        ),
    ]


def get_security_target(name: str) -> SecurityTargetSpec:
    for target in default_security_targets():
        if target.name == name:
            return target
    raise KeyError(f"Unknown security target: {name}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_security_target(name: str) -> SecurityResolvedTarget:
    spec = get_security_target(name)
    artifact_dir = Path(spec.artifact_dir)
    release_catalog_path = artifact_dir / "catalog.json"
    resolved_root = artifact_dir

    if spec.profile is not None and release_catalog_path.exists():
        profile_dir = artifact_dir / "profiles" / spec.profile
        if profile_dir.exists() or profile_dir.is_symlink():
            resolved_root = profile_dir
        else:
            catalog = _load_json(release_catalog_path)
            profiles = {item.get("name"): item for item in catalog.get("profiles", []) if isinstance(item, dict)}
            profile_payload = profiles.get(spec.profile)
            if profile_payload and isinstance(profile_payload.get("source_dir"), str):
                resolved_root = Path(profile_payload["source_dir"])

    server_dir: Path | None = None
    if (resolved_root / "server").exists():
        server_dir = resolved_root / "server"

    client_secret_path: Path | None = None
    for candidate in [
        resolved_root / "client" / "client_secret.pt",
        resolved_root / "client_secret.pt",
    ]:
        if candidate.exists():
            client_secret_path = candidate
            break

    metadata_path: Path | None = None
    for candidate in [
        resolved_root / "stage_i_metadata.json",
        resolved_root / "metadata.json",
    ]:
        if candidate.exists():
            metadata_path = candidate
            break

    manifest_path: Path | None = None
    if (resolved_root / "manifest.json").exists():
        manifest_path = resolved_root / "manifest.json"

    return SecurityResolvedTarget(
        name=spec.name,
        stage=spec.stage,
        track=spec.track,
        artifact_dir=str(artifact_dir),
        resolved_root_dir=str(resolved_root),
        profile=spec.profile,
        variant=spec.variant,
        server_dir=str(server_dir) if server_dir is not None else None,
        client_secret_path=str(client_secret_path) if client_secret_path is not None else None,
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        release_catalog_path=str(release_catalog_path) if release_catalog_path.exists() else None,
    )


def security_targets_payload() -> dict[str, Any]:
    return {
        "format": "qwen_security_targets_v1",
        "targets": [item.to_dict() for item in default_security_targets()],
        "resolved_targets": [resolve_security_target(item.name).to_dict() for item in default_security_targets()],
    }
