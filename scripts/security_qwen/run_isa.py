from __future__ import annotations

import argparse

from _shared import run_template_builder
from src.security_qwen import build_isa_template


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--observable-type", default="hidden_state")
    parser.add_argument("--observable-layer", default="planned")


def kwargs_from_args(args: argparse.Namespace) -> dict:
    return {
        "observable_type": args.observable_type,
        "observable_layer": args.observable_layer,
    }


if __name__ == "__main__":
    run_template_builder("isa", build_isa_template, extra_args=add_args, kwargs_from_args=kwargs_from_args)
