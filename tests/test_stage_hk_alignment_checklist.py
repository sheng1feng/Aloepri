from pathlib import Path


def test_alignment_checklist_mentions_attention_and_norm() -> None:
    text = Path("docs/阶段H-K论文对齐检查表.md").read_text(encoding="utf-8")
    assert "attention" in text
    assert "norm" in text
