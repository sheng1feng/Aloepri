# Qwen 安全总报告

## 1. 报告目标

本文档用于汇总当前 `Qwen` 安全评测工作的整体结论，回答四个问题：

1. 你已经复现了哪些安全相关内容；
2. 当前复现结果里最主要的安全不足是什么；
3. 与原始论文的安全结论相比，有哪些差异；
4. 当前仓库在安全分析维度上处于什么完成度。

同时，本轮还必须保留一个口径边界：

> 现有 Gate 1-6 的数值性安全结论，主要描述的是 **legacy conservative deployment line**，也就是旧版 `Stage J / Stage K` 所代表的保守部署线；它们不能自动外推到当前已经补齐骨架、但尚未完成 Gate 级复跑的 redesigned `Stage H-K` 线上。

---

## 2. 当前复现范围

### 2.1 方法复现范围

当前仓库的 `Qwen` 主线可以拆成三部分理解：

- `Stage A-G`：共同前置方法主链
- legacy `Stage H/I/J/K`：当前 Gate 1-6 安全结论的主要评测对象
- redesigned `Stage H/I/J/K`：远端最新代码已经补齐阶段定义、导出脚本、桥接/标准权重证明、物化与 release 包装，但尚未完成新版 Gate 级复跑

### 2.2 安全评测范围

当前 `Qwen` 安全评测已完成以下 Gate：

- `Gate 1 / VMA`
- `Gate 2 / IMA`
- `Gate 3 / ISA`
- `Gate 4 / TFMA + SDA`
- `Gate 5 / 参数-安全性联合扫描`
- `Gate 6 / 增强点验证`

当前尚未完成的主要论文攻击项：

- `IA`

### 2.3 当前结论适用边界

当前已完成的 Gate 1-6，主要围绕以下对象展开：

- `stage_h_full_obfuscated`
- `stage_j_full_square`
- `stage_j_full_square_tiny_a`
- `stage_k_release`

其中：

- `Stage H` 更接近研究线
- `Stage J / Stage K` 更接近旧版保守部署线

因此，下面的结果总览应优先理解为对 legacy 线的总结；对 redesigned 线的最终判断，仍需要等新版 Gate 复跑完成后才能下结论。

---

## 3. 当前结果总览

### 3.1 Gate 1 / VMA

结论：

- `Stage H` 明显优于 `Stage J / K`
- `Stage J / K` 在更强 `projection-enhanced VMA` 下接近 `Stage A` baseline

含义：

- 部署线存在显著的结构恢复风险；
- 研究线与部署线的主要安全差异首先体现在 `VMA` 这类结构型攻击上。

### 3.2 Gate 2 / IMA

结论：

- 最小线性训练式反演器已经足以高度恢复各类工件的 embedding 语义；
- `Stage H / J / K` 都没有在这条攻击线上形成足够强的隔离。

含义：

- 当前系统不只是“结构容易被恢复”；
- 表示空间本身也高度可学习。

### 3.3 Gate 3 / ISA

结论：

- `hidden_state` 上有一定恢复能力，但整体不强；
- `attention_score` 当前最小基线几乎无效；
- `Stage H` 在 `hidden_state` 上仍优于 `Stage J / K`。

含义：

- 当前部署态 observable 攻击不是主风险；
- 至少在当前最小实现下，它弱于 `VMA / IMA`。

### 3.4 Gate 4 / TFMA + SDA

结论：

- 频率攻击在 `Stage A / H / J / K` 上几乎无差异；
- 风险主要由 token permutation 决定；
- 先验越强，攻击越强。

含义：

- hidden-space 混淆几乎不会自然缓解长期在线统计泄露；
- 频率攻击是 token-level 固有风险。

### 3.5 Gate 5 / 参数扫描

结论：

- 在可接受正确性区间内，调大 `alpha_e / alpha_h` 几乎不能显著降低核心攻击风险；
- 真正显著降低 `VMA` 的强噪声点会让正确性不可接受。

含义：

- 当前问题不是“参数还没调好”；
- 而是“方法路径本身的安全上限”。

### 3.6 Gate 6 / 增强点验证

结论：

- 轻量敏感 token 定向加噪几乎没有收益；
- 极端定向加噪能降低敏感 `VMA`，但正确性受损且 `IMA` 仍很强。

含义：

- 简单补丁不足以解决核心问题；
- 当前方法需要更结构性的增强。

---

## 4. 当前最主要的安全不足

把全部结果汇总后，当前最重要的安全不足有 4 条：

### 4.1 部署线结构恢复风险高

`Stage J / K` 在更强 `VMA` 下显著弱于 `Stage H`。

### 4.2 embedding 语义恢复风险高

`IMA` 在几乎所有工件上都表现出高恢复率。

### 4.3 频率泄露天然存在

`TFMA / SDA` 几乎不区分 `Stage A/H/J/K`，说明这部分风险主要由 token permutation 决定。

### 4.4 调参与轻量增强的收益有限

- Gate 5 说明在可接受正确性区间内调参难以换来安全收益；
- Gate 6 说明简单定向加噪不足以同时压住 `VMA + IMA`。

---

## 5. 为什么部署线弱于 Stage H

当前最合理的根因解释是：

1. `Stage H` 更接近研究线，保留了更复杂的 KeyMat / attention 结构扰动；
2. `Stage J / Stage K` 为标准 HF 部署收缩成了更规整的单项式 hidden transform；
3. 这种规整性让 `q / k / v / gate / up` 多组权重更容易被联合利用；
4. embed/head 噪声不足以掩盖内部层共同结构；
5. `Stage K` 只是包装层，基本继承 `Stage J` 的安全属性。

更详细分析见：

- `docs/qwen_security/部署线弱于StageH的原因分析.md`
- `docs/AloePri 论文中的部署适配机制整理.md`

---

## 6. 与原始论文的差异

### 6.1 一致的部分

- 攻击面定义一致：
  - 结构恢复
  - 训练式反演
  - token-frequency 利用
- 论文的威胁模型与你的复现实验方向是对齐的。

### 6.2 不一致的部分

当前你的结果比论文主张的安全性更悲观：

- 论文声称 AloePri 对 `VMA / IMA / IA / ISA` 都有较强抗性；
- 但你在本地 `Qwen2.5-0.5B-Instruct` 复现中看到：
  - `Stage J / K` 在 `VMA` 下很脆；
  - 所有线在最小 `IMA` 下都很脆；
  - `TFMA / SDA` 在有先验时会明显增强。

### 6.3 不能简单对论文下“对/错”结论

因为当前仍有显著差异：

- 论文主结果基于更大模型；
- 论文环境更接近生产集群；
- 你的攻击实现是“最小可闭环基线”，不是论文原版全部细节；
- redesigned `Stage H-K` 虽然已经有新的部署骨架，但还没完成同口径 Gate 复跑。

因此更准确的说法是：

> 当前本地小模型复现并未复现出论文中同等级别的安全优势，反而暴露出更强攻击面；而 redesigned 线是否能收回这部分差距，仍需要后续复跑结果验证。

---

## 7. 当前完成度判断

### 7.1 代码

- 安全线代码已经具备独立的模块、脚本、汇总与目标解析逻辑；
- redesigned `Stage H-K` 相关的桥接、物化、审计和包装代码已经并入主线。

### 7.2 文档

- `docs/qwen_security/` 已形成索引、Gate 报告、推进看板、简报与总报告；
- 根 README 已补充安全评测入口；
- 论文部署适配机制的纠偏依据已单独整理成文。

### 7.3 测试

- 当前已有 summary、resolve、redesign 目标和文档存在性的专项测试；
- 今天补齐了 Gate 5 / Gate 6 工件 resolve 的覆盖；
- 完整全量回归仍建议在新版 Gate 复跑前单独执行一次。

### 7.4 主要缺口

- `IA` 尚未实现；
- redesigned `Stage H-K` 尚未重跑 Gate 级安全评测；
- `scripts/run_aloepri_modular.py` 的独立改动目前仍只有脚本级改动，没有自动化回归覆盖。

---

## 8. 当前建议

如果继续推进安全线，建议优先级如下：

1. 先在 redesigned `Stage H-K` 线上复挂安全评测目标；
2. 优先重跑 `Gate 1 / VMA` 与 `Gate 2 / IMA`；
3. 再决定是否需要重跑 `Gate 3-6`；
4. 如果继续做方法增强，应优先参考论文部署适配约束，避免只在 `alpha_e / alpha_h` 这类参数上做局部微调。

---

## 9. 一句话总结

> 当前仓库已经完成了 `Qwen` 路线首轮 Gate 1-6 安全闭环，并补齐了 redesigned `Stage H-K` 的阶段骨架与部署适配分析；但就现有数值结果而言，最悲观的结论仍主要落在 **legacy conservative deployment line** 上，即部署线 `Stage J / Stage K` 在结构恢复上明显弱于研究线 `Stage H`，各条线在最小训练式反演下也高度可恢复。下一步最关键的工作，是把同口径安全评测真正挂到 redesigned 线上。
