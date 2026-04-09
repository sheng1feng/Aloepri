from __future__ import annotations

from _shared import run_template_builder
from src.security_qwen import build_ima_template


if __name__ == "__main__":
    run_template_builder("ima", build_ima_template)
