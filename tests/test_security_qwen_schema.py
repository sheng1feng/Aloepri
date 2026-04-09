from src.security_qwen import (
    build_isa_template,
    build_vma_template,
    get_security_target,
    validate_security_eval_payload,
)


def test_vma_template_matches_phase0_schema() -> None:
    target = get_security_target("stage_j_stable_reference").to_target()
    payload = build_vma_template(target)
    ok, errors = validate_security_eval_payload(payload)
    assert ok is True
    assert errors == []
    assert payload["attack"] == "vma"
    assert payload["summary"]["status"] == "planned"


def test_isa_template_carries_observable_fields() -> None:
    target = get_security_target("stage_h_full_obfuscated").to_target()
    payload = build_isa_template(target, observable_type="kv_cache", observable_layer="layer_7")
    ok, errors = validate_security_eval_payload(payload)
    assert ok is True
    assert errors == []
    assert payload["metrics"]["observable_type"] == "kv_cache"
    assert payload["metrics"]["observable_layer"] == "layer_7"
