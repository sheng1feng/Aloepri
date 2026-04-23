# 阶段 I：部署约束验证报告

> Canonical note: 本文档只回答当前 `Stage I` 的状态，不承担全局主线说明。Qwen 唯一主线入口见 [docs/论文一致最终部署主线.md](论文一致最终部署主线.md)。

## 1. 新阶段目标

新版 `Stage I` 的职责是验证：

> 新版 `Stage H` 中保留下来的可部署混淆表达，是否真的仍处在标准推理框架的约束边界内。

这里的重点不是单独打通 `Stage A` 入口，而是明确：

- 哪些表达仍可写回标准权重
- 哪些表达只差工程物化
- 哪些表达会直接触碰 HF / vLLM / SGLang 的运行时假设

## 2. 当前边界

当前 `Stage I` deployability matrix 明确保留以下约束：

- 运行时仍保持标准 Transformer 计算图
- 不引入新的在线自定义算子
- 兼容目标面向：
  - `transformers`
  - `vllm`
  - `sglang`

## 3. 与 legacy Stage I 的关系

旧版 `Stage I` 更偏：

- `Stage A` 标准 HF/vLLM 入口打通
- feasibility probe
- Phase 2 shape/materialization 阻塞定位

这些内容仍保留，但不再是新版 `Stage I` 的完整定义。

legacy 参考：

- `docs/阶段I_vLLM接入计划.md`
- `docs/阶段I_vLLM复现报告.md`
