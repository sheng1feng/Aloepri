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
    main_path = "docs/论文一致最终部署主线.md"
    for path in [
        "docs/阶段H_Qwen可部署混淆表达重构报告.md",
        "docs/阶段I_部署约束验证报告.md",
        "docs/阶段J_论文一致部署路线说明.md",
        "docs/阶段K_Qwen交付包装报告.md",
    ]:
        text = Path(path).read_text(encoding="utf-8")
        assert "不承担全局主线说明" in text
        assert main_path in text
