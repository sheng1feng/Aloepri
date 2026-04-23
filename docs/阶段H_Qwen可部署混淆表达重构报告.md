# 阶段 H：Qwen 可部署混淆表达重构报告

> Canonical note: 本文档只回答当前 `Stage H` 的状态，不承担全局主线说明。Qwen 唯一主线入口见 [docs/论文一致最终部署主线.md](论文一致最终部署主线.md)。

## 1. 新阶段目标

新版 `Stage H` 的职责不再是单纯做 attention 静态化收敛和噪声定标，而是：

> 明确哪些论文中的复杂混淆表达，仍然可以在 **不改变标准 Transformer 运行时计算图** 的前提下，被吸收到 Qwen 标准组件权重中。

这个阶段的输出重点不是 full-model export，而是一个可以被后续 `Stage I / J / K` 继续使用的**可部署参数表达清单**。

## 2. 当前保留的可部署表达

当前代码线中，至少有以下几类表达已经可以作为新版 `Stage H` 的基础：

- `embedding/head`
  - token permutation
  - key-matrix side transform
  - noise terms
- `attention`
  - block permutation
  - head/group diversity
  - RoPE side rotation and scaling profile
- `FFN`
  - component-specific transform
  - per-layer diversity
- `norm`
  - kappa correction
  - offline fusion preference

## 3. 与 legacy Stage H 的关系

旧版 `Stage H` 仍然重要，但它的语义更适合作为：

- research-line deployment-oriented artifact stabilization
- attention staticization evidence
- noise calibration evidence

而不是新版 `Stage H` 的完整定义。

对应 legacy 文档：

- `docs/阶段H-Attention静态化与噪声定标报告.md`
- `docs/阶段H_混淆模型部署说明.md`

## 4. 本阶段闭环标准

新版 `Stage H` 闭环要求：

- 有 canonical inventory builder
- 有可导出的 JSON 结果
- 有新的阶段报告
- 旧版 `Stage H` 文档被显式标记为 legacy 参考

这一步完成后，后续 `Stage I` 才有明确的对象去验证“这些表达在 HF / vLLM / SGLang 约束下是否可部署”。
