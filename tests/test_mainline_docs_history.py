from pathlib import Path


def test_active_root_docs_exist() -> None:
    assert Path("docs/复现主线总览.md").exists()
    assert Path("docs/论文一致最终部署主线.md").exists()
    assert Path("docs/Llama-3.2-3B最终部署主线.md").exists()


def test_history_subtrees_exist() -> None:
    for path in [
        "docs/history/qwen",
        "docs/history/llama",
        "docs/history/security",
        "docs/history/shared",
    ]:
        assert Path(path).is_dir()


def test_qwen_security_tree_moves_under_history() -> None:
    assert not Path("docs/qwen_security").exists()
    assert Path("docs/history/security/qwen_security/README.md").exists()


def test_old_shared_entry_docs_move_under_history() -> None:
    assert not Path("docs/仓库技术文档.md").exists()
    assert not Path("docs/Qwen与Llama复现阶段区分说明.md").exists()
    assert Path("docs/history/shared/仓库技术文档.md").exists()
    assert Path("docs/history/shared/Qwen与Llama复现阶段区分说明.md").exists()


def test_old_llama_entry_docs_move_under_history() -> None:
    assert not Path("docs/Llama-3.2-3B快速使用说明.md").exists()
    assert not Path("docs/Llama-3.2-3B云端验证说明.md").exists()
    assert not Path("docs/Llama-3.2-3B本机改造与云验证计划.md").exists()
    assert Path("docs/history/llama/Llama-3.2-3B快速使用说明.md").exists()
    assert Path("docs/history/llama/Llama-3.2-3B云端验证说明.md").exists()
    assert Path("docs/history/llama/Llama-3.2-3B本机改造与云验证计划.md").exists()


def test_repository_overview_has_stage_map_and_paper_gap_matrix() -> None:
    text = Path("docs/复现主线总览.md").read_text(encoding="utf-8")
    assert "## 2. 整个复现流程摘要" in text
    assert "## 5. 部署线与论文差异对照表" in text
    assert "| 维度 | 原始论文目标 | Qwen 当前主线 | Llama 当前主线 |" in text
    assert "`artifacts/stage_k_release`" in text
    assert "`artifacts/stage_k_llama_release`" in text
    assert "`paper_consistent`" in text
    assert "`stable_reference / tiny_a`" in text
