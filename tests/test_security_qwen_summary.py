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
    text = Path("docs/history/security/qwen_security/Qwen安全总报告.md").read_text(encoding="utf-8")
    assert "legacy conservative deployment line" in text
