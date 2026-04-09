from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.security_qwen.schema import validate_security_eval_payload


def iter_security_eval_files(root: str | Path = "outputs/security_qwen") -> Iterable[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*.json") if path.is_file())


def load_security_eval_payloads(root: str | Path = "outputs/security_qwen") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_payloads: list[dict[str, Any]] = []
    invalid_payloads: list[dict[str, Any]] = []

    for path in iter_security_eval_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid_payloads.append({"path": str(path), "error": f"json_load_failed: {exc}"})
            continue

        if payload.get("format") != "qwen_security_eval_v1":
            continue

        ok, errors = validate_security_eval_payload(payload)
        if ok:
            valid_payloads.append({"path": str(path), "payload": payload})
        else:
            invalid_payloads.append({"path": str(path), "error": "; ".join(errors)})

    return valid_payloads, invalid_payloads


def security_summary_payload(root: str | Path = "outputs/security_qwen") -> dict[str, Any]:
    valid_payloads, invalid_payloads = load_security_eval_payloads(root)
    entries: list[dict[str, Any]] = []
    by_attack: dict[str, int] = {}

    for item in valid_payloads:
        payload = item["payload"]
        attack = payload["attack"]
        target = payload["target"]
        summary = payload.get("summary", {})
        by_attack[attack] = by_attack.get(attack, 0) + 1
        entries.append(
            {
                "path": item["path"],
                "attack": attack,
                "stage": target.get("stage"),
                "profile": target.get("profile"),
                "artifact_dir": target.get("artifact_dir"),
                "primary_metric_name": summary.get("primary_metric_name"),
                "primary_metric_value": summary.get("primary_metric_value"),
                "risk_level": summary.get("risk_level"),
                "status": summary.get("status"),
            }
        )

    return {
        "format": "qwen_security_summary_v1",
        "root": str(root),
        "valid_result_count": len(valid_payloads),
        "invalid_result_count": len(invalid_payloads),
        "by_attack": by_attack,
        "entries": entries,
        "invalid_entries": invalid_payloads,
    }
