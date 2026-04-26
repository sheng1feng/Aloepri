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
    assert "artifacts/stage_k_release/" in text


def test_readme_points_to_repository_and_dual_mainlines() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "docs/复现主线总览.md" in text
    assert "docs/论文一致最终部署主线.md" in text
    assert "docs/Llama-3.2-3B最终部署主线.md" in text


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
    index_text = Path("docs/history/security/qwen_security/README.md").read_text(encoding="utf-8")
    board_text = Path("docs/history/security/qwen_security/推进看板.md").read_text(encoding="utf-8")
    assert "不是 Qwen 部署主线入口" in index_text
    assert "仅跟踪安全子域" in board_text
    canonical_link = "[docs/论文一致最终部署主线.md](../../../论文一致最终部署主线.md)"
    assert canonical_link in index_text
    assert canonical_link in board_text


def test_mainline_doc_next_steps_are_real_remaining_work() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 7\. 下一步顺序(.*?)## 8\.", text, re.S)
    assert section is not None
    next_steps = section.group(1)

    assert "唯一 release 面" in next_steps
    assert "correctness" in next_steps
    assert "`VMA / IMA / ISA`" in next_steps
    assert "`default` / `reference`" in next_steps

    assert "固定唯一主线文档与阶段主报告" not in next_steps
    assert "合并 Stage-J 冗余文档" not in next_steps
    assert "移除 legacy / bridge / redesign 并列叙事" not in next_steps


def test_stage_j_docs_use_paper_consistent_candidate_as_active_target() -> None:
    route_text = Path("docs/阶段J_论文一致部署路线说明.md").read_text(encoding="utf-8")
    main_text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    readme_text = Path("README.md").read_text(encoding="utf-8")
    assert "artifacts/stage_j_qwen_paper_consistent" in route_text
    assert "artifacts/stage_j_qwen_paper_consistent" in readme_text
    assert "outputs/stage_j/paper_consistent/completion_summary.json" in main_text
    assert "outputs/stage_j/paper_consistent/correctness_regression.json" in main_text


def test_stage_j_docs_demote_historical_bridge_to_auxiliary_evidence() -> None:
    route_text = Path("docs/阶段J_论文一致部署路线说明.md").read_text(encoding="utf-8")
    assert "历史中间线" in route_text
    assert "artifacts/stage_j_qwen_redesign_standard" in route_text


def test_stage_k_docs_use_paper_consistent_release_surface() -> None:
    stage_k_text = Path("docs/阶段K_Qwen交付包装报告.md").read_text(encoding="utf-8")
    main_text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    readme_text = Path("README.md").read_text(encoding="utf-8")
    assert "artifacts/stage_k_release" in stage_k_text
    assert "artifacts/stage_j_qwen_paper_consistent" in stage_k_text
    assert "outputs/stage_j/paper_consistent/correctness_regression.json" in stage_k_text
    assert "`default`" in stage_k_text
    assert "`reference`" in stage_k_text
    assert "artifacts/stage_k_release" in main_text
    assert "outputs/stage_j/paper_consistent/correctness_regression.json" in main_text
    assert "artifacts/stage_k_release" in readme_text


def test_llama_active_docs_are_rooted_under_llama_mainline() -> None:
    text = Path("docs/Llama-3.2-3B最终部署主线.md").read_text(encoding="utf-8")
    assert "docs/Llama-3.2-3B标准形状恢复报告.md" in text
    assert "docs/Llama-3.2-3B噪声定标与StageK推进说明.md" in text
    assert "docs/Llama-3.2-3B客户端与Server使用说明.md" in text
    assert "docs/history/llama/Llama-3.2-3B快速使用说明.md" in text


def test_security_docs_move_to_history_tree() -> None:
    text = Path("docs/复现主线总览.md").read_text(encoding="utf-8")
    assert "docs/history/security/qwen_security/" in text


def test_mainline_doc_no_longer_lists_stage_k_cutover_as_remaining_work() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    section = re.search(r"## 7\. 下一步顺序(.*?)## 8\.", text, re.S)
    assert section is not None
    next_steps = section.group(1)
    assert "唯一 release 面" in next_steps
    assert "correctness" in next_steps
    assert "`VMA / IMA / ISA`" in next_steps
    assert "将 `Stage K` 切换到最终论文一致线" not in next_steps


def test_qwen_mainline_doc_has_explicit_paper_gap_section() -> None:
    text = Path("docs/论文一致最终部署主线.md").read_text(encoding="utf-8")
    assert "## 6. 与原始论文的当前差异" in text
    assert "不能表述为“已经与论文完全等价”" in text
    assert "`paper_consistent`" in text
    assert "`VMA / IMA / ISA`" in text


def test_llama_mainline_doc_lists_release_evidence_inputs() -> None:
    text = Path("docs/Llama-3.2-3B最终部署主线.md").read_text(encoding="utf-8")
    assert "## 9. 当前证据入口" in text
    assert "outputs/stage_j_llama/real_remote_validation.json" in text
    assert "outputs/stage_j_llama/real_tiny_a_remote_validation.json" in text
    assert "artifacts/stage_k_llama_release/catalog.json" in text
