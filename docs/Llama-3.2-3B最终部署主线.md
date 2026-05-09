# Llama-3.2-3B 最终部署主线

## 1. 唯一目标

当前 Llama-3.2-3B 文档只保留一条主线：

> **Llama-3.2-3B 的最终可交付部署线**

这条主线当前以独立 `Stage K` release 为最终交付面：

- `artifacts/stage_k_llama_release`

当前 Llama 主线明显落后于 Qwen。

## 2. 当前状态总览

当前 Llama 主线已经完成：

- `LlamaArchitectureAdapter` 接入
- 本机 smoke、Stage I、Stage J 标准形状导出链路成立
- 真实 `RTX 4090` 上的 correctness 验证
- 真实噪声定标
- Llama 专属 `Stage K release`

这意味着当前 Llama 主线已经从“结构接入实验”推进到“真实可交付部署线”，但它仍主要停留在 `deployability / correctness` 交付层，不是论文同态收口。

## 3. 当前 release 语义

当前 `artifacts/stage_k_llama_release` 使用两个 profile：

- `stable_reference`
- `tiny_a`

它们的语义是：

- `stable_reference`：零噪声 correctness / 调试基线
- `tiny_a`：当前推荐的非零噪声默认交付 profile

当前推荐 profile 为：

- `tiny_a`

## 4. 当前活跃文档

Llama 主线当前只保留这 3 份活跃支撑文档：

- [docs/Llama-3.2-3B标准形状恢复报告.md](Llama-3.2-3B标准形状恢复报告.md)
- [docs/Llama-3.2-3B噪声定标与StageK推进说明.md](Llama-3.2-3B噪声定标与StageK推进说明.md)
- [docs/Llama-3.2-3B客户端与Server使用说明.md](Llama-3.2-3B客户端与Server使用说明.md)

它们分别回答：

- 标准形状恢复与 correctness 证据
- 噪声定标与 release profile 形成
- 实际 client/server 使用方式

## 5. 从接入到交付的主线摘要

### 5.1 结构接入阶段

- 让仓库识别 `LlamaForCausalLM`
- 打通 `llama_decoder` 结构下的主干导出路径

### 5.2 本机验证阶段

- 用 mock / smoke 方式验证 Stage I 与 Stage J 的标准形状导出链路
- 证明这条线在本机 CPU 上已可导出、可回归、可为云端验证准备工件

### 5.3 云端真实验证阶段

- 在真实 `Llama-3.2-3B` 与 `RTX 4090` 环境下完成 correctness 验证
- 确认 Stage I 与 Stage J 都达到可用交付标准

### 5.4 噪声定标与 release 阶段

- 对真实 3B 工件进行噪声定标
- 形成 `stable_reference` / `tiny_a` 两个活跃 profile
- 导出独立的 `Stage K` release

## 6. 与 Qwen 的关键差异

Llama 与 Qwen 采用相同的高层复现逻辑，但当前活跃语义并不相同。

- Qwen 当前根口径是 `paper_consistent`
- Llama 当前根口径是自身 release profile：`stable_reference / tiny_a`
- Qwen 当前活跃问题集中在最终 release 面的复跑与安全复证
- Llama 当前更像“release 已成型，但论文同口径安全叙事仍未完全补齐”

因此两条线是并行活跃主线，不是同一条线的不同名字；其中 Llama 当前明显落后于 Qwen。

## 7. 与原始论文的差异

当前 Llama 主线已经形成了可部署工件，但仍与原始论文理想目标存在差异。

### 已对齐的部分

- 标准 HF 工件形态
- client/server 闭环
- correctness 层面的真实验证
- 非零噪声工作点与 release packaging

### 尚未完全对齐的部分

- 尚未形成与 Qwen `paper_consistent` 对应的论文一致根口径
- 尚未完成与论文同强度、同口径的安全攻击评测闭环
- 当前更强调 `deployability / correctness`，而不是已经完成所有论文层面的最终证明

## 8. 当前仍未完成的关键项

- 以当前 `Stage K` release 为口径补齐更完整的安全评测
- 视需要补齐更强部署后端或更广复跑口径
- 如后续要与 Qwen 更强对齐，再决定是否引入统一的论文一致语义根

## 9. 当前证据入口

- `stable_reference` correctness：`outputs/stage_j_llama/real_remote_validation.json`
- `tiny_a` correctness：`outputs/stage_j_llama/real_tiny_a_remote_validation.json`
- 噪声定标结果：`outputs/stage_j_llama/real_noise_calibration.json`
- `Stage K` release catalog：`artifacts/stage_k_llama_release/catalog.json`
- 活跃 release 面：`artifacts/stage_k_llama_release`

## 10. 历史文档

以下旧文档将统一迁入 `docs/history/llama/`：

- `docs/history/llama/Llama-3.2-3B快速使用说明.md`
- `docs/history/llama/Llama-3.2-3B云端验证说明.md`
- `docs/history/llama/Llama-3.2-3B本机改造与云验证计划.md`

它们只保留历史或辅助说明价值，不再作为当前主线入口。
