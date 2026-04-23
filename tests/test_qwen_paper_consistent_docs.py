import re
from pathlib import Path


def test_canonical_qwen_deployment_doc_exists() -> None:
    assert Path("docs/论文一致最终部署主线.md").exists()


def test_readme_uses_single_qwen_deployment_entry() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "docs/论文一致最终部署主线.md" in text
    assert "Qwen Stage H-K redesign" not in text
    assert "docs/阶段H-K重构迁移说明.md" not in text


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
