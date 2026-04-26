# 阶段 H-K 重构迁移说明

## 1. 目标

这份文档用于说明：

- 旧版 `Stage H/I/J/K` 是什么；
- 新版 `Stage H/I/J/K` 准备如何重定义；
- 当前仓库里的哪些工件仍属于 legacy 参考线。

本次重构不改变 `Stage A-G`，只重排 Qwen 的 `Stage H-K` 语义与推进路径。

## 2. 旧版映射

- 旧 `Stage H`
  - 更偏 research-line artifact stabilization
  - 重点是 attention 静态化、噪声定标、部署说明
- 旧 `Stage I`
  - 更偏 `Stage A` 标准 HF/vLLM 入口打通
  - 重点是 feasibility probe
- 旧 `Stage J`
  - 更偏 conservative standard-shape deployment line
  - 重点是 square monomial / non-expanding full-layer 恢复
- 旧 `Stage K`
  - 更偏对旧 `Stage J` 工件做 release packaging

## 3. 新版主线

- 新 `Stage H`
  - Qwen 可部署混淆表达重构
- 新 `Stage I`
  - 部署约束验证
- 新 `Stage J`
  - Qwen 全模型部署物化
- 新 `Stage K`
  - 交付包装与运行时收口

## 4. legacy 参考对象

下列文档和工件仍然保留，但不再作为新版 `H-K` 的唯一目标定义：

- `docs/阶段H-Attention静态化与噪声定标报告.md`
- `docs/阶段H_混淆模型部署说明.md`
- `docs/阶段I_vLLM接入计划.md`
- `docs/阶段I_vLLM复现报告.md`
- `docs/阶段J_标准形状前缀恢复报告.md`
- `docs/阶段J_标准形状噪声定标报告.md`
- `docs/阶段K_标准形状交付包装报告.md`

## 5. 重构原则

- `可部署` 指的是运行时仍保持标准 Transformer 计算图
- `可部署` 不等于退化成单一全局 hidden transform
- 只要复杂扰动能吸收到标准组件权重中，就不应被过早删除
- 每个阶段必须单独完成文档、代码、测试、验证闭环后再推进下一阶段
