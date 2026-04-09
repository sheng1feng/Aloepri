from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security_qwen import get_security_target


def build_template_parser(attack: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Export a Phase 0 template payload for {attack}.")
    parser.add_argument("--target", required=True, help="Security target name, e.g. stage_j_stable_reference.")
    parser.add_argument("--output-path", default="", help="Optional JSON output path.")
    return parser


def write_payload(payload: dict, output_path: str) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved payload to {path}")
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def resolve_target(target_name: str):
    return get_security_target(target_name).to_target()


def run_template_builder(
    attack: str,
    builder: Callable[..., dict],
    *,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    kwargs_from_args: Callable[[argparse.Namespace], dict] | None = None,
) -> None:
    parser = build_template_parser(attack)
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args()
    target = resolve_target(args.target)
    kwargs = kwargs_from_args(args) if kwargs_from_args is not None else {}
    payload = builder(target, **kwargs)
    write_payload(payload, args.output_path)
