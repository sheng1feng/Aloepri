# AloePri 论文中的部署适配机制整理

**——用于指导现有部署线纠正**

## 1. 文档目的

本文档聚焦回答一个非常具体的问题：

> 原始 AloePri 论文是如何在保留混淆能力的同时，适配 vLLM / SGLang 等标准推理框架的？

这个问题的意义不在于复述论文安全性结论，而在于为当前项目的部署线纠偏提供依据。当前项目已经发现，部署线 `Stage J / K` 相比研究线 `Stage H` 在结构恢复攻击下明显更脆，因此需要重新检查：**论文中的“可部署”到底意味着什么，哪些混淆表达是被保留下来的，哪些约束是真实存在的，哪些地方是我们当前实现中过度收缩了。**

---

## 2. 论文对“部署适配”的总体定位

论文从一开始就没有把“部署适配”当成后置工程工作，而是把它列为方法设计的核心约束之一。论文明确提出，工业场景中的隐私保护 LLM 推理必须同时满足三类约束：

1. 精度和效率损失要尽量小；
2. 能运行在异构、遗留 xPU 集群上；
3. **必须兼容现有 LLM 基础设施，以复用 vLLM、SGLang 等框架的工程优化。**

论文图 1 进一步把工作流画得很明确：客户端先做 **offline model obfuscation**，将混淆后的模型部署到服务器；服务器侧继续使用现有 inference engine，例如 **vLLM / SGLang**；在线阶段客户端只负责 prompt obfuscation 和 response de-obfuscation。也就是说，论文的目标不是重新发明一套推理框架，而是让**混淆后的模型仍然能作为一个普通模型，被标准推理框架直接加载和运行**。

论文在摘要、引言和贡献部分反复强调这一点：AloePri **does not change the model structure**，并且可以 **direct integration into mainstream inference frameworks (e.g., vLLM and SGLang)**。实验部分也明确写出，服务端部署使用的是 **vLLM 0.9.1 + CUDA 12.4**。

**结论**：论文不是“摆脱部署约束”，而是**在方法层面主动服从部署约束**，并把可部署性内化到混淆设计里。

---

## 3. 论文中的部署适配原则

从论文实现可以抽象出 4 条最关键的部署适配原则。

### 3.1 原则一：不改模型结构，只改参数表达

论文明确把 AloePri 的一个核心特点表述为：

> 不改变模型结构，并与现有 LLM infrastructures 兼容。

这句话的真实含义不是“什么都不能变”，而是：

* 不新增在线推理模块；
* 不修改 Transformer 基本计算图；
* 不要求推理框架认识一种新的 attention/FFN/norm 算子；
* 不改 auto-regressive generation 的接口形式；
* 不要求 server 和 client 在在线阶段做多轮交互。

换句话说，**混淆必须尽量体现在权重改写上，而不是体现在运行时推理图变化上。**

这也是论文为什么坚持把主要复杂性放在 **offline model obfuscation**，而把在线阶段尽量保持为“普通模型 + 普通 prompt 推理”的原因。

---

### 3.2 原则二：复杂性前移到离线阶段，在线推理路径尽量接近明文推理

论文图 1、图 2 和第 5.1 节都说明了 AloePri 的整体思路：

* **离线阶段**：客户端生成 secret permutation、噪声、key matrix、inverse key matrix，并对 embedding、head、attention、FFN、norm 权重做统一混淆；
* **在线阶段**：客户端只做 token-level prompt obfuscation，服务器仅接收 obfuscated prompt 和 obfuscated model 做推理，最后客户端再做输出去混淆。

论文还专门强调：

> its inference pipeline remains nearly identical to plaintext inference.

这意味着 AloePri 的部署适配核心不是“让框架支持新结构”，而是：

> **先把混淆预编译进模型参数，然后把混淆后的参数当作普通 checkpoint 交给推理框架。**

这也是它能继续复用 KV Cache、并行矩阵计算、P/D disaggregation 等工程优化的根本原因。

---

### 3.3 原则三：混淆必须通过标准组件的等价权重变换来表达

论文第 5.2 节最大的特点是：
对每一个组件，论文都不是新增结构，而是把混淆写成**标准权重的变换**。

#### 3.3.1 Embedding / Head

论文先对 embedding 与 head 加噪，再施加 token permutation，以及 key / inverse-key matrix 变换，最终得到：

* ( \widetilde{W}*{embed} = \Pi W^**{embed} \hat{P}_{embed} )
* ( \widetilde{W}*{head} = \hat{Q}*{head} W^*_{head} \Pi^T )

也就是说，部署到服务器上的仍然是**标准 embedding 层和标准 LM head 层**，只是参数值已经变成 obfuscated form。

#### 3.3.2 Attention

论文 attention obfuscation 看起来复杂，但其输出仍然是标准的：

* ( \widetilde{W_q}, \widetilde{W_k}, \widetilde{W_v}, \widetilde{W_o} )

其复杂性体现在这些权重是通过以下机制离线构造出来的：

* intra-head transformation
* RoPE 对应的旋转矩阵 ( \hat{R}_{qk} )
* scaling 矩阵 ( \hat{H}_{qk} )
* block-wise permutation
* inter-head permutation：( \tau_{kv}, \tau_{group} )
* key / inverse-key matrix
* value/output 侧的随机可逆矩阵 ( \hat{U}_{vo} )

但在部署态，服务器仍然只看到一个标准 attention 层。

论文还明确说，这套方法可以直接应用于 **MHA、MQA、GQA**；对于 **MLA**，则对其低秩权重做另一组可逆变换。重点在于：**计算图保持原 attention 范式不变，只是权重被离线改写。**

#### 3.3.3 FFN

FFN 也是同样思路。论文没有改 FFN 结构，而是通过：

* permutation ( \hat{Z}_{ffn} )
* scaling ( \hat{H}_{ffn} )
* key / inverse-key matrix

去构造新的 ( \widetilde{W}*{gate}, \widetilde{W}*{up}, \widetilde{W}_{down} )。
MoE 场景下，额外对 router 进行 permutation 以保证 expert 选择仍然正确。

部署时，server 依然运行的是普通 dense FFN 或普通 MoE FFN。

#### 3.3.4 RMSNorm

LayerNorm / RMSNorm 是论文里最能体现“部署适配”思维的部分之一。论文并没有要求框架支持一个新的 norm 算子，而是：

* 用 ( \kappa = \mathbb{E}[|x\hat{P}|/|x|] ) 修正 obfuscated normalization 偏差；
* 再把修正后的 RMSNorm 与相邻线性层融合。

论文原文甚至明确说：

> The weights of the linear layer (W_{norm}) can be merged into the layer adjacent to the RMSNorm layer before applying weight obfuscation.

这说明他们非常在意：**尽量不要让部署态多出新的非标准算子。**

---

### 3.4 原则四：允许“参数层面的复杂性”，但不允许“运行时结构层面的复杂性”

这是理解论文部署适配的关键。

AloePri 并不是一个“简单统一变换”方案。论文在参数层面保留了相当多复杂混淆表达：

* token permutation
* embedding/head 噪声
* left-right key matrix transformation
* attention 的 head/block/group 扰动
* RoPE 配套旋转与 scaling
* FFN 中间维 permutation / scaling
* norm 修正。

但这些复杂性都被限制在一个边界内：

> **最终部署出来的模型，仍然必须表现为标准 Transformer 组件的权重版本。**

也就是说，论文允许：

* 参数值变复杂，
* 参数来源变复杂，
* 权重构造过程变复杂，

但不允许：

* 在线推理图新增非标准模块，
* attention kernel 运行语义发生根本变化，
* 框架必须额外理解一种私有数据流。

这就是 AloePri 看起来“既复杂又能部署”的根本原因。

---

## 4. 论文并非完全无视部署约束，而是在约束内保留了尽可能多的混淆表达

这一点必须说清楚。

你现在容易产生一个误解：论文似乎做到了“保留 Stage H 级复杂混淆，同时完全不受部署约束”。
其实不是。

论文真实做法是：

* **承认部署约束存在**；
* **在这些约束内，尽量把复杂混淆吸收进标准参数表达**；
* 只保留那些不会破坏标准推理图和工程入口的表达。

所以 AloePri 的成就不是“没有约束”，而是：

> 在不改模型结构、不破坏推理框架兼容性的前提下，仍然把足够多的混淆信息保留进了参数中。

这和当前项目中“为了标准交付，过度收缩成统一、规整、跨层复用的单项式 hidden transform”并不是一回事。

---

## 5. 对当前部署线纠偏最重要的论文启示

下面这部分是给你现有 Stage J/K 最有用的。

### 5.1 纠偏启示一：不能把“标准部署”误读成“只允许全局统一变换”

论文 clearly 不是这么做的。

虽然 AloePri 要求兼容 vLLM / SGLang，但论文并没有因此退化成：

* 单一全局 permutation
* 单一全局 sign flip
* 单一全局 scale
* 所有层共享同一套隐藏变换

相反，它在 attention 和 FFN 中仍然保留了：

* per-head
* per-block
* per-group
* per-side
* per-component

的复杂扰动。

**纠偏含义**：
当前部署线如果把“可部署”收缩理解成“所有模块共用一个简单、统一、跨层复用的 monomial transform”，那很可能是对论文部署适配的过度简化。

---

### 5.2 纠偏启示二：部署线不应只在 embed/head 上保留混淆，内部层也必须保留结构扰动

论文里 embedding/head 的 permutation 和 noise 只是入口和出口保护。真正为了防止服务器从内部层恢复 token permutation，论文还专门对 attention、FFN、norm 做了成体系的参数混淆。

尤其是 6.5 的消融实验，论文明确指出：

* 只有 noise 时，ISA 对 attention score 和 hidden state 都很危险；
* 加上 key matrix 后，hidden state 才显著安全；
* 再加上 head & block permutation 后，attention score 才真正被压住。

**纠偏含义**：
当前部署线如果把复杂扰动主要压缩掉，只剩 embed/head 层做保护，那么结构恢复和内部态攻击必然会重新变强。这与你现在 Gate 1 / Gate 3 的观察高度一致。

---

### 5.3 纠偏启示三：论文的可部署表达仍然允许 attention 内部复杂扰动

这是最关键的纠偏点之一。

论文的 attention 部分不是“统一 hidden transform 共轭”就结束了，而是额外保留了：

* ( \hat{R}_{qk} )：RoPE 兼容的旋转
* ( \hat{H}_{qk} )：query/key scaling
* ( \hat{Z}_{block} )：block-wise permutation
* ( \tau_{kv} )：KV 头级别 permutation
* ( \tau_{group} )：group 内 query/output permutation。

这说明论文认为：

> **attention 侧复杂结构扰动，本身就是部署态安全性的重要来源。**

**纠偏含义**：
如果你当前部署线为了“标准 shape”把这些 attention 侧复杂表达尽量删掉，只保留最简单的参数融合，那么就已经和论文的“可部署版 AloePri”有明显偏差。

---

### 5.4 纠偏启示四：论文允许 key matrix 带扩展维度，不等于必须退回最保守原始 hidden 空间

论文的 Key Matrix Generation 明确引入了 expansion size (h)，并说明通过 (h) 可以构造无限多组满足 ( \hat{P}\hat{Q}=I ) 的 key / inverse-key matrices。

后面的效率实验还专门研究了不同 expansion size (h) 对 TTFT/TPOT 的影响，结论是：增大 (h) 只会带来轻微推理时延增加。

**纠偏含义**：
论文所谓“兼容部署”并不等于“完全禁止扩展表达”。
如果你当前部署线把所有非原始 hidden-size 的表达都当成不可部署内容并一刀切删掉，那么可能删掉了论文中本来被保留下来的安全来源。

---

### 5.5 纠偏启示五：RMSNorm 不能简单回退成普通权重置换，必须保留论文的数值修正逻辑

论文专门为 RMSNorm 设计了 ( \kappa ) 修正，用来抵消变换后 norm 统计偏差，并说明这部分可以 fuse 到相邻层。

这说明 norm 不是一个“可忽略小细节”，而是部署适配中必须谨慎处理的数值稳定环节。

**纠偏含义**：
如果当前部署线只是为了标准 checkpoint 导出，简单把 norm 也视为普通线性权重重排对象，而没有保留论文对应的修正逻辑，那么很可能在数值正确性和安全表达两边都出现偏差。

---

## 6. 当前部署线应如何对照论文做修正

基于上面的分析，建议把现有部署线纠偏拆成 4 项核查。

### 6.1 核查一：部署线是否过度统一化

检查点：

* 是否大量模块共享同一个全局 hidden transform
* 是否多个 decoder 层都重复暴露同一套秘密结构
* 是否 attention / FFN / norm 最终都退化成统一单项式共轭改写

如果答案是“是”，那么这和论文部署思路并不一致。论文兼容部署，但并没有把所有混淆都压成一个跨层统一的简单变换。

---

### 6.2 核查二：attention 复杂扰动是否被丢失

检查点：

* 是否仍保留 block-wise permutation
* 是否仍保留 head permutation / group permutation
* 是否仍保留 RoPE 对应旋转与 scaling
* 是否这些结构只是“分析阶段存在”，而没有真正写入部署线权重

如果这些都缺失，那么当前部署线已经弱化了论文最关键的部署态安全来源之一。

---

### 6.3 核查三：FFN 与 router 是否只做了最弱的可逆变换

检查点：

* gate/up/down 是否仍有独立 permutation + scaling
* MoE router 是否保持 expert-level permutation
* 是否为了标准导出把这些扰动收缩掉了

如果 FFN 侧也被大幅规整化，那么当前部署线会更容易被多模块联合恢复。

---

### 6.4 核查四：norm 修正与 key matrix 机制是否真实落地

检查点：

* 是否实现了论文里的 ( \kappa ) 修正
* 是否 key / inverse-key matrices 的表达在部署态还存在
* 是否 expansion size (h) 被过度压缩为 0 或形式保留、实质删除
* 是否所谓“标准 shape”实际上让 key matrix 的安全意义大量流失

如果这些点没有保留，那么你当前部署线虽然更“像普通 checkpoint”，但已经不是论文语境下的 AloePri 部署适配了。

---

## 7. 可直接指导当前项目的结论

把全文压缩成一句最有操作性的判断：

> **原始论文并不是通过放弃混淆表达来适配 vLLM，而是通过把混淆设计成 embedding、head、attention、FFN、norm 等标准组件的离线参数改写，使 obfuscated model 在运行时仍表现为标准 Transformer；因此，论文的部署适配并不等于“统一、规整、跨层复用的简单 hidden transform”，而是“在不改模型结构和在线推理图的前提下，尽量保留参数层面的复杂扰动”。**

这句话对应到你当前项目，就是：

> 当前部署线真正需要纠正的，不是“还不够标准”，而是“为了标准交付，可能把论文中本来允许保留的复杂参数扰动也过度压缩掉了”。

---

## 8. 建议的下一步落地动作

建议按下面顺序推进纠偏：

### 动作 1：做一张“论文部署表达 vs 当前 Stage J/K 表达”的逐项对照表

对照项至少包括：

* embed/head noise + permutation + key matrix
* attention 中的 RoPE 旋转 / scaling / block perm / head perm / group perm
* FFN 的 permutation / scaling / key matrix
* norm 的 ( \kappa ) 修正和 fuse
* key matrix 的 expansion size (h)

这一步的目标是确认：当前部署线到底在哪些点上**少做了**，哪些点只是**做法不同但等价**。

### 动作 2：优先恢复 attention 侧可部署扰动

从论文证据看，部署态安全差距最容易首先出现在 attention 侧。
优先检查是否能在不破坏标准框架入口的前提下，补回：

* block permutation
* head/group permutation
* RoPE 对应旋转与 scaling。

### 动作 3：检查部署线是否错误地把“标准结构”理解成“禁止局部多样性”

论文要求的是模型结构标准，不是所有层都必须共享同一秘密变换。
需要重点验证：

* per-layer diversity 是否被不必要删除
* 多层是否在重复暴露同一全局 transform

### 动作 4：最后再看导出和框架兼容性

不要先以“导出更干净”为目标再压缩混淆，而应该先判断：

* 哪些复杂扰动其实仍可吸收到标准参数里
* 哪些才是真的会破坏 vLLM / SGLang 运行假设

---

## 9. 一句话总结

> AloePri 的部署适配本质不是“去掉复杂混淆”，而是“把复杂混淆尽可能吸收到标准 Transformer 组件的参数表达中”，从而让 obfuscated model 在在线阶段仍可作为标准模型交给 vLLM / SGLang 运行；因此，现有部署线若为了标准化而过度收缩成统一、规整、跨层复用的简单变换，就已经偏离了论文的可部署设计思想。
