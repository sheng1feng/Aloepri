# Qwen 安全总报告

## 1. 报告目标

本文档用于汇总当前 `Qwen` 安全评测工作的整体结论，并补充一个在本轮阶段重构后必须明确的新口径：

> 现有 Gate 1-6 的安全结论，描述的主要是 **legacy conservative deployment line**，也就是旧版 `Stage J / Stage K` 所代表的保守部署线；它们不能自动外推到当前正在重构的新版 `Stage H-K` 线上。

## 2. 当前范围

### 2.1 已完成的方法主线

当前仓库已经有两条需要区分的后段语义：

- `Stage A-G`
  - 作为共同前置方法主链
- legacy `Stage H/I/J/K`
  - 当前安全 Gate 结果的主要评测对象来源
- redesigned `Stage H/I/J/K`
  - 当前已经完成阶段定义、文档、代码入口与包装骨架，但尚未完成新一轮安全 Gate 复跑

### 2.2 已完成的安全 Gate

当前 `Qwen` 安全评测已完成：

- `Gate 1 / VMA`
- `Gate 2 / IMA`
- `Gate 3 / ISA`
- `Gate 4 / TFMA + SDA`
- `Gate 5 / 参数-安全性联合扫描`
- `Gate 6 / 增强点验证`

当前尚未完成的主要论文攻击项：

- `IA`

## 3. 当前安全结论对应的是哪条线

本轮必须明确区分：

### 3.1 legacy conservative deployment line

当前已完成的 Gate 1-6，主要围绕以下对象展开：

- `stage_h_full_obfuscated`
- `stage_j_full_square`
- `stage_j_full_square_tiny_a`
- `stage_k_release`

其中：

- `Stage H` 更接近研究线
- `Stage J / Stage K` 更接近旧版保守部署线

### 3.2 redesigned Stage H-K line

当前仓库已经建立了新版阶段骨架：

- `Stage H`
  - 可部署混淆表达重构
- `Stage I`
  - 部署约束验证
- `Stage J`
  - `artifacts/stage_j_qwen_redesign`
- `Stage K`
  - release catalog 已开始指向 redesigned `Stage J`

但这条 redesign 线尚未重新跑 Gate 级攻击，因此：

> 当前安全总报告里的数值结论，仍然应视为对 **legacy conservative deployment line** 的总结，而不是对 redesign 线的最终判决。

## 4. 当前结果总览（legacy 线）

### 4.1 Gate 1 / VMA

- `Stage H` 明显优于 `Stage J / K`
- `Stage J / K` 在更强 `projection-enhanced VMA` 下接近 `Stage A` baseline

含义：

- 部署线存在显著结构恢复风险
- 研究线与部署线的主要安全差异首先体现在 `VMA`

### 4.2 Gate 2 / IMA

- 最小线性训练式反演器已经足以高度恢复各类工件的 embedding 语义
- `Stage H / J / K` 都没有形成足够强的隔离

含义：

- 当前系统不只是“结构容易被恢复”
- 表示空间本身也高度可学习

### 4.3 Gate 3 / ISA

- `hidden_state` 上有一定恢复能力，但整体不强
- `attention_score` 当前最小基线几乎无效
- `Stage H` 在 `hidden_state` 上仍优于 `Stage J / K`

### 4.4 Gate 4 / TFMA + SDA

- 频率攻击在 `Stage A / H / J / K` 上几乎无差异
- 风险主要由 token permutation 决定
- 先验越强，攻击越强

### 4.5 Gate 5 / Gate 6

- 在可接受正确性区间内，调大 `alpha_e / alpha_h` 几乎不能显著降低核心风险
- 简单的敏感 token 定向加噪不足以同时压低 `VMA / IMA`

## 5. 当前最重要的项目判断

截至目前，最稳妥的项目级判断仍然是：

1. 当前最强风险来自 `VMA` 和 `IMA`
2. 当前最糟糕的结论主要发生在 legacy `Stage J / Stage K` 这条 conservative deployment line 上
3. 当前问题更像是“旧部署线的结构安全上限”，而不是“参数没调好”

## 6. 当前完成度

### 代码

- 安全线代码、脚本、测试已经形成独立目录
- redesign `Stage H-K` 也已经建立新的代码骨架与交付入口

### 文档

- `docs/qwen_security/` 已形成 Gate 报告、计划、看板与总报告
- `docs/阶段H-K重构迁移说明.md` 已建立新旧阶段语义区分

### 缺口

- `IA` 尚未实现
- redesign `Stage H-K` 尚未重跑 Gate 级安全评测

## 7. 下一步建议

1. 先在 redesign `Stage H-K` 线上复挂安全评测目标
2. 优先重跑 `Gate 1 / VMA` 与 `Gate 2 / IMA`
3. 再决定是否需要重跑 `Gate 3-6`

## 8. 一句话总结

> 当前 `Qwen` 安全线已经完成首轮 Gate 1-6 闭环，但这些结论主要刻画的是 **legacy conservative deployment line**；随着新版 `Stage H-K` 重构落地，下一步最关键的工作不是继续解释旧结论，而是把安全评测重新挂到 redesign 线，验证新的 deployable line 是否真的优于旧 `Stage J / Stage K`。 
