from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SecurityEvalTarget:
    stage: str
    artifact_dir: str
    profile: str | None
    model_family: str
    variant: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_security_eval_payload(
    *,
    attack: str,
    target: SecurityEvalTarget | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload_target = target.to_dict() if isinstance(target, SecurityEvalTarget) else dict(target)
    return {
        "format": "qwen_security_eval_v1",
        "attack": attack,
        "target": payload_target,
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
        "summary": dict(summary or {}),
        "artifacts": dict(artifacts or {}),
    }


def validate_security_eval_payload(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if payload.get("format") != "qwen_security_eval_v1":
        errors.append("format must be 'qwen_security_eval_v1'")

    attack = payload.get("attack")
    if not isinstance(attack, str) or not attack:
        errors.append("attack must be a non-empty string")

    target = payload.get("target")
    if not isinstance(target, Mapping):
        errors.append("target must be a mapping")
    else:
        for field_name in ["stage", "artifact_dir", "model_family", "variant"]:
            value = target.get(field_name)
            if not isinstance(value, str) or not value:
                errors.append(f"target.{field_name} must be a non-empty string")
        profile = target.get("profile")
        if profile is not None and not isinstance(profile, str):
            errors.append("target.profile must be a string or null")

    for field_name in ["config", "metrics", "summary", "artifacts"]:
        if not isinstance(payload.get(field_name), Mapping):
            errors.append(f"{field_name} must be a mapping")

    return len(errors) == 0, errors
