from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security_qwen import (
    build_ia_template,
    build_ima_template,
    build_isa_template,
    build_sda_template,
    build_tfma_template,
    build_vma_template,
    get_security_target,
)


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cases = [
        ("vma", "stage_j_stable_reference", build_vma_template(get_security_target("stage_j_stable_reference").to_target())),
        ("ia", "stage_j_tiny_a", build_ia_template(get_security_target("stage_j_tiny_a").to_target())),
        ("ima", "stage_a_standard", build_ima_template(get_security_target("stage_a_standard").to_target())),
        (
            "isa",
            "stage_h_full_obfuscated",
            build_isa_template(
                get_security_target("stage_h_full_obfuscated").to_target(),
                observable_type="hidden_state",
                observable_layer="layer_0",
            ),
        ),
        (
            "tfma",
            "stage_k_reference",
            build_tfma_template(
                get_security_target("stage_k_reference").to_target(),
                knowledge_setting="zero_knowledge",
            ),
        ),
        (
            "sda",
            "stage_k_default",
            build_sda_template(
                get_security_target("stage_k_default").to_target(),
                knowledge_setting="domain_aware",
            ),
        ),
    ]

    for attack, name, payload in cases:
        output_path = Path(f"outputs/security_qwen/{attack}/{name}.template.json")
        _write_payload(output_path, payload)
        print(f"Saved template: {output_path}")


if __name__ == "__main__":
    main()
