import re
from pathlib import Path


def test_canonical_qwen_deployment_doc_exists() -> None:
    assert Path("docs/论文一致最终部署主线.md").exists()


def test_readme_uses_single_qwen_deployment_entry() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "docs/论文一致最终部署主线.md" in text
    assert "Qwen Stage H-K redesign" not in text
    assert "docs/阶段H-K重构迁移说明.md" not in text
    assert "Qwen 主线：A–K 完整复现" not in text
    assert "J：standard-shape full-layer 恢复" not in text
    assert "artifacts/stage_j_full_square/" not in text
    assert "artifacts/stage_j_full_square_tiny_a/" not in text
    assert "artifacts/stage_k_release/" not in text


def test_readme_uses_canonical_qwen_header_once() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert text.count("## Qwen 论文一致最终部署主线") == 1


def test_canonical_stage_reports_are_stage_local() -> None:
    for path in [
        "docs/阶段H_Qwen可部署混淆表达重构报告.md",
        "docs/阶段I_部署约束验证报告.md",
        "docs/阶段J_论文一致部署路线说明.md",
        "docs/阶段K_Qwen交付包装报告.md",
    ]:
        text = Path(path).read_text(encoding="utf-8")
        assert "不承担全局主线说明" in text
        assert re.search(
            r"\[[^\]]*论文一致最终部署主线\.md[^\]]*\]\((?:\./)?(?:docs/)?论文一致最终部署主线\.md\)",
            text,
        )


def test_stage_j_route_doc_absorbs_bridge_and_weight_proof() -> None:
    text = Path("docs/阶段J_论文一致部署路线说明.md").read_text(encoding="utf-8")
    assert "标准权重可见性证明" in text
    assert "standard-visible bridge line" in text
    assert "buffered redesign line" in text


def test_redundant_stage_j_docs_removed() -> None:
    assert not Path("docs/阶段J_Qwen全模型部署物化报告.md").exists()
    assert not Path("docs/阶段J_标准可见桥接导出报告.md").exists()
    assert not Path("docs/阶段J_标准权重证明报告.md").exists()


def test_security_docs_are_subordinate_to_main_line() -> None:
    index_text = Path("docs/qwen_security/README.md").read_text(encoding="utf-8")
    board_text = Path("docs/qwen_security/推进看板.md").read_text(encoding="utf-8")
    assert "不是 Qwen 部署主线入口" in index_text
    assert "仅跟踪安全子域" in board_text
    canonical_link = "[docs/论文一致最终部署主线.md](../论文一致最终部署主线.md)"
    assert canonical_link in index_text
    assert canonical_link in board_text


def test_mainline_doc_next_steps_are_real_remaining_work() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 6\. 下一步顺序(.*?)## 7\.", text, re.S)
    assert section is not None
    next_steps = section.group(1)

    assert "paper-consistent final `Stage J` 产物" in next_steps
    assert "attention / FFN / norm" in next_steps
    assert "export-visible" in next_steps
    assert "`Stage K`" in next_steps
    assert "最终论文一致线" in next_steps
    assert "correctness / `VMA / IMA / ISA`" in next_steps

    assert "固定唯一主线文档与阶段主报告" not in next_steps
    assert "合并 Stage-J 冗余文档" not in next_steps
    assert "移除 legacy / bridge / redesign 并列叙事" not in next_steps
