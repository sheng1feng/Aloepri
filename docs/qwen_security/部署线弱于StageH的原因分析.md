# 部署线为何显著弱于 Stage H：原因分析

## 1. 问题定义

当前安全复现中，一个最明确的现象是：

- `Stage H` 研究线在 `VMA` 下明显更稳；
- `Stage J / K` 部署线在更强 `projection-enhanced VMA` 下明显更脆；
- `Stage K` 与对应 `Stage J` profile 基本一致。

因此需要回答：

> 为什么部署线 `Stage J / K` 会显著弱于 `Stage H`？

本文档的目标不是重复列出结果，而是解释这些结果背后的**结构性原因**。

---

## 2. 先说结论

更准确的说法不是：

- `Stage J / K` “实现坏了”

而是：

- `Stage J / K` 为了标准 HF / 标准形状部署，**主动牺牲了部分研究线中的混淆表达力**；
- 这使得部署线保留了更多可被 `VMA` 利用的线性结构关系；
- 因而一旦攻击者联合利用 `q / k / v / gate / up` 等多组权重，部署线就更容易暴露出可恢复的共同结构。

一句话概括：

> `Stage H` 更接近“研究型复杂混淆表达”；  
> `Stage J / K` 更接近“可部署的规整共轭变换”；  
> 正是这种“规整性”让部署线更容易被结构恢复攻击利用。

---

## 3. 现象回顾

## 3.1 Gate 1 的关键信号

### Direct-only

- `Stage A` 近乎完全可恢复；
- `Stage H` 恢复率很低；
- `Stage J / K` 位于中间。

### Projection-enhanced

- `Stage A` 仍近乎完全可恢复；
- `Stage J / K` 迅速逼近 `Stage A`；
- `Stage H` 虽然也上升，但仍明显更低。

这说明：

1. `Stage J / K` 的问题不是只出在 embedding/head；
2. 真正的风险来自**内部层权重关系**；
3. 一旦攻击者能联合利用多组投影权重，部署线的保护强度会急剧下降。

## 3.2 Gate 2~4 为什么不能直接解释这个差异

- `Gate 2 / IMA`
  - 对所有线都很强；
  - 它告诉我们表示空间泄露严重，但并不能解释“为什么部署线比研究线更弱”。

- `Gate 3 / ISA`
  - hidden-state 有一定差异，但不是主导项；
  - attention-score 最小基线几乎无效。

- `Gate 4 / TFMA + SDA`
  - 几乎完全不区分 `Stage A/H/J/K`；
  - 说明频率攻击主要由 token permutation 决定，不是这里的核心差异来源。

所以：

> 真正把 `Stage H` 和 `Stage J / K` 拉开差距的，是 **Gate 1 / VMA 这类结构恢复攻击**。

---

## 4. 根因 1：研究线与部署线的目标不同

这是最根本的一点。

### `Stage H`

`Stage H` 的定位更接近：

- 研究复现
- KeyMat 路线
- attention 静态化后的复杂混淆表达

它关注的是：

- 混淆表达力
- 方法链可解释性
- 靠近论文原始设计的内部结构

### `Stage J / K`

`Stage J / K` 的定位更接近：

- 标准 HF tensor shape
- 标准 checkpoint
- release 包装
- 可部署性

它关注的是：

- 能否保持标准 shape
- 能否导出标准工件
- 能否被现有推理入口加载

因此部署线从一开始就不是在优化“最强混淆”，而是在优化“混淆 + 标准部署兼容性”。

这意味着：

> 部署线的安全让步并不是偶然现象，而是其设计目标的一部分副作用。

---

## 5. 根因 2：Stage J 使用了更规整、更统一的全局单项式变换

`Stage J` 的核心 hidden transform 是：

- permutation
- sign flip
- 可选全局缩放

也就是一类 **square monomial transform**。

这类变换的优点是：

- 不改变标准 hidden size
- 易于写回标准 HF checkpoint
- 易于在层间保持功能正确

但坏处同样明显：

- 变换形式太规整；
- 在各层中复用同一个全局结构；
- 很多投影权重都被同一种规则共轭。

从攻击者角度看，这意味着：

> `q_proj / k_proj / v_proj / gate_proj / up_proj / down_proj / norm`  
> 不再是多个独立的混淆对象，而是多个**共同指向同一隐藏变换**的观察窗口。

这正是 `projection-enhanced VMA` 会突然变强的原因。

---

## 6. 根因 3：Stage J 的内部层主要是确定性共轭改写，而不是额外随机扰动

`Stage J` 里对层的适配，本质上是：

- 输入侧线性层乘同一个 transform
- 输出侧线性层乘同一个 inverse / transpose
- RMSNorm 做对应的 weight permutation

这类写法很适合保证：

- `Block(xP) ≈ Block(x)P`

但它并不会主动打散：

- head 内部结构
- block 内部结构
- FFN 中间维的层间随机关系

同时，`Stage J` 的噪声主要还是集中在：

- embedding
- head

而不是系统性地渗透到所有内部层的参数表达里。

因此会出现一个很典型的现象：

- 只看 embedding/head 时，部署线似乎还有一定保护；
- 一旦引入内部层的投影权重，攻击者马上能把这些关系拼起来。

---

## 7. 根因 4：Stage H 保留了更多复杂 attention/FFN 扰动

这是 `Stage H` 更强的真正技术来源。

在研究线里，attention 侧不仅仅是一个统一变换，而是引入了更复杂的结构：

- `R_qk`
- `H_qk`
- `Z_block`
- `tau_kv`
- `tau_group`

这些操作会同时影响：

- head 内特征顺序
- block 对齐方式
- kv head 排列
- query group 排列

因此对于攻击者来说，`Stage H` 的 attention 权重不只是“被同一个全局矩阵变换了”，而是：

- 还被 head/block/group 级别的结构扰动进一步打散了。

从你的 `source attribution` 结果看：

- 对 `Stage J`，`q / k / v / gate` 任一族单独都很强；
- 对 `Stage H`，`q/k/v` 并不强，反而 `gate/up` 更值得警惕。

这恰恰说明：

> `Stage H` 对 attention 结构的保护更充分；  
> `Stage J` 则让 attention/FFN 多组投影重新暴露出可联合利用的规律。

---

## 8. 根因 5：Stage H 还保留了扩维 KeyMat 路线，而 Stage J 必须回到标准形状

这是另一个关键差异。

### `Stage H`

研究线仍运行在：

- 扩维 hidden representation
- metric-matrix style RMSNorm
- restore matrix
- 静态化 attention 表达

这意味着服务器侧看到的权重与内部表示，本身就不再是标准模型空间里的简单线性关系。

### `Stage J`

部署线必须满足：

- hidden size 不扩维
- RMSNorm 保持标准算子形态
- q/k/v/o 与 FFN 都保持标准 shape

所以它必须把复杂结构“压缩回”标准模型参数空间。

这个压缩过程的直接代价是：

- 原来由扩维和 metric structure 隐藏掉的一些结构，
- 重新变成了标准张量空间里可比对、可排序、可匹配的关系。

也就是说：

> `Stage J` 不是把 `Stage H` 的安全性无损搬进标准 shape；  
> 它是用更保守的表达，换取部署可行性。

---

## 9. 根因 6：Stage J 的层间“多样性”不足

你的实验里还有一个很重要的信号：

- `Stage J` 的单层中后层就已经很危险；
- 多层联合只是把这个危险再放大。

这说明：

- 部署线中，不同层看到的不是完全独立的秘密结构；
- 它们更像是“同一种秘密结构在多层上的重复显现”。

而研究线中，由于：

- per-layer FFN transform
- per-layer attention config
- 更复杂的 block/head/group 扰动

不同层之间的可拼接性更弱。

从攻击者视角看，这意味着：

- `Stage J`：层与层是“投票伙伴”
- `Stage H`：层与层更像“部分不兼容的多个视角”

这就是为什么多层投票对 `Stage J` 提升更大。

---

## 10. 为什么 Stage K 没有更差

这点反而最简单。

`Stage K` 做的主要是：

- catalog
- deployment contract
- profile 化包装
- 统一推理入口

它并不重新定义模型混淆数学。

因此 `Stage K` 与对应 `Stage J` profile 在安全性上几乎一致，是完全合理的结果。

换句话说：

> `Stage K` 不是新的安全层；  
> 它只是把 `Stage J` 的安全属性原样打包出来。

---

## 11. 这是不是实现 bug？

当前更合理的判断是：

- **不是主要由实现 bug 导致**
- 而是**设计约束导致的系统性差异**

当然，仍然可能存在一些实现细节让差距被放大，例如：

1. 部署线内部层噪声太少
2. 某些 head/block permutation 没能等价迁移到标准形状路线
3. 同一个全局 transform 在过多模块中重复暴露

但即使没有这些细节问题，下面这件事仍然成立：

> 只要标准形状路线必须回到“统一、规整、非扩维”的参数表达，它就天然更容易暴露可被结构恢复利用的关系。

---

## 12. 当前最可能的直接原因排序

如果要按“贡献度”排序，我会给出如下判断：

### 第一原因

**部署线把复杂混淆收缩成统一的 square monomial transform。**

这是最核心的原因。

### 第二原因

**内部层缺少 Stage H 那种复杂 head/block/group 级扰动。**

这直接导致 `q/k/v/gate/up` 重新变成高价值攻击源。

### 第三原因

**部署线的 embed/head 噪声不足以掩盖内部层的共同结构。**

这解释了为什么 Gate 5 里小幅调参没用。

### 第四原因

**层间多样性不够，导致多层投票效果极强。**

这解释了为什么 `projection-enhanced` 会急剧放大恢复率。

---

## 13. 你接下来该怎么验证这个判断

如果要继续做“原因定位”而不是直接进入新方法设计，我建议做 4 个最值钱的验证：

### 验证 A：逐权重族贡献排序

目标：

- 定量比较 `q / k / v / gate / up` 的单独贡献

意义：

- 判断到底是 attention 侧还是 FFN 侧更先泄露。

### 验证 B：逐层贡献排序

目标：

- 比较 `layer_0 / middle / last` 单层与组合层

意义：

- 判断危险是否集中在中后层。

### 验证 C：给 Stage J 加入 layer-specific transform

目标：

- 测试“同一全局 transform 复用”是不是主因

意义：

- 如果 per-layer transform 后 `VMA` 显著下降，就基本坐实这一点。

### 验证 D：在标准形状路线中补回 head/block permutation

目标：

- 测试 attention 结构扰动的缺失是不是主因

意义：

- 如果补回后 `q/k/v` 贡献显著下降，就说明当前问题主要在 attention 侧。

---

## 14. 一句话结论

> 部署线 `Stage J / K` 显著弱于 `Stage H`，主要不是因为实现坏了，而是因为它为了标准 HF 部署，把研究线中更复杂、更多样、带扩维与 head/block/group 扰动的混淆表达，收缩成了统一、规整、非扩维的单项式 hidden 变换；这虽然保住了功能和可部署性，但也重新暴露了跨层、跨模块可被统一恢复的结构关系，从而显著放大了 `VMA` 类结构恢复攻击面。 
