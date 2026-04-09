from __future__ import annotations

import argparse

from _shared import run_template_builder
from src.security_qwen import build_sda_template


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knowledge-setting", default="zero_knowledge")


def kwargs_from_args(args: argparse.Namespace) -> dict:
    return {"knowledge_setting": args.knowledge_setting}


if __name__ == "__main__":
    run_template_builder("sda", build_sda_template, extra_args=add_args, kwargs_from_args=kwargs_from_args)
