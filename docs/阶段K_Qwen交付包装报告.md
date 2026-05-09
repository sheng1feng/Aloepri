# 阶段 K：Qwen 交付包装报告

> Canonical note: 本文档只回答当前 `Stage K` 的状态，不承担全局主线说明。Qwen 唯一主线入口见 [docs/论文一致最终部署主线.md](论文一致最终部署主线.md)。

> 历史阶段报告与旧审计材料将统一归入 `docs/history/qwen/`。

## 1. 当前唯一发布面

当前 `Stage K` 只保留一个活跃发布目录：

- `artifacts/stage_k_release`

它包装的唯一源工件是：

- `artifacts/stage_j_qwen_paper_consistent`

## 2. 当前 profile 语义

当前 `Stage K` 使用两个 profile 名称：

- `default`
- `reference`

两者当前都指向同一个 `paper_consistent` `Stage J` 工件，但承担不同入口语义：

- `default`：默认交付入口
- `reference`：审计与证据入口

## 3. 交付职责

当前 `Stage K` 的职责是：

- 提供 release catalog
- 固定 client/server contract
- 提供统一推理入口
- 把 `Stage J` 的论文一致候选工件收口成唯一发布面

当前 `Stage K` 的活跃 correctness 证据只认 release-surface 自身导出的结果文件：

- `outputs/stage_k_release/correctness/default.json`
- `outputs/stage_k_release/correctness/reference.json`
- `outputs/stage_k_release/correctness_summary.json`

`outputs/stage_j/paper_consistent/completion_summary.json` 只表示 `Stage J` 是否达到 export-visible completion，不再作为 `Stage K` 的 correctness 证据解释。

当前 `Stage K` release 面已经补齐当前主线安全结果：

- `outputs/security_qwen/vma/stage_k_default.json`
- `outputs/security_qwen/vma/stage_k_reference.json`
- `outputs/security_qwen/ima/stage_k_default.paper_like.json`
- `outputs/security_qwen/ima/stage_k_reference.paper_like.json`
- `outputs/security_qwen/isa/stage_k_default.hidden_state.json`
- `outputs/security_qwen/isa/stage_k_reference.hidden_state.json`
- `outputs/security_qwen/isa/stage_k_default.attention_score.json`
- `outputs/security_qwen/isa/stage_k_reference.attention_score.json`

其中当前结论是：

- `VMA`：低风险
- `ISA(hidden_state / attention_score)`：低风险
- `IMA`：低风险

当前 `Stage K` 的 `IMA` 结论只认 `paper_like` 结果，也就是：

- `scripts/security_qwen/run_ima.py --mode paper_like`
- `outputs/security_qwen/ima/stage_k_default.paper_like.json`
- `outputs/security_qwen/ima/stage_k_reference.paper_like.json`

## 4. 与历史 Stage K 的关系

旧版 `Stage K` 命名与旧 profile 语义仅作为历史证据保留，不再代表当前唯一论文部署线发布面。
