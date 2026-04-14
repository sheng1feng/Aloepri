from pathlib import Path


def test_stage_hk_redesign_docs_exist() -> None:
    assert Path("docs/阶段H-K重构迁移说明.md").exists()


def test_readme_mentions_redesigned_qwen_stage_hk_line() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Qwen Stage H-K redesign" in text
