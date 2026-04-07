# AloePri 技术报告梳理与复现方案

本文档基于 `docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf` 整理，目标是帮助后续开展 **Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation**（文中方法名为 **AloePri**）的工程复现。

---

## 1. 论文一句话概括

这篇论文提出了一种面向工业 LMaaS 场景的隐私保护推理方法 **AloePri**：不是只对输入做混淆，而是**同时对输入数据和模型参数做协同变换**（covariant obfuscation），让模型在“混淆空间”里继续近似正确推理，同时尽量不破坏现有推理框架、硬件部署方式和在线吞吐。

---

## 2. 论文试图解决什么问题

### 2.1 背景

在云端 LLM 推理场景中，客户端必须把 prompt 发给远端服务，服务端还会生成包含隐私信息的输出。因此，隐私暴露既发生在：

- 输入 prompt 传输与处理阶段；
- 模型中间状态暴露阶段；
- 输出文本返回阶段。

### 2.2 论文提出的工业化约束

论文认为，真正可落地的隐私保护 LLM 推理方案要同时满足三类约束：

1. **精度与效率约束**
   - 不能造成明显精度下降；
   - 不能引入显著在线延迟或吞吐损失。

2. **硬件兼容约束**
   - 要能运行在异构、存量 xPU 集群上；
   - 不能依赖大规模硬件改造。

3. **软件兼容约束**
   - 要尽量兼容现有推理基础设施；
   - 能复用 vLLM、SGLang 等系统已有的 KV Cache、并行推理、服务化工程优化。

### 2.3 论文对现有方法的判断

作者认为现有方法大致分三类，但都无法同时满足上述三类约束：

- **文本/Token 替换类**
  - 优点：轻量；
  - 问题：只保护局部敏感 token，语义易受损，生成任务精度下降明显。

- **Embedding 变换/分割推理类**
  - 优点：保护粒度更细；
  - 问题：客户端负担重，且常暴露 embedding / hidden state，容易被 inversion attack 利用。

- **密码学或 TEE 类**
  - 优点：理论安全性强；
  - 问题：要么太慢，要么依赖硬件改造，难以直接复用现有 LLM 推理基础设施。

---

## 3. 论文核心思想

### 3.1 关键词：协变混淆（Covariant Obfuscation）

论文的核心不是“只把数据变掉”，而是把 **数据、模型参数、输出空间** 一起映射到一个新的混淆空间中。

作者把一个推理函数写成：

- `f: X × Θ -> Y`
  - `X`：输入空间
  - `Θ`：参数空间
  - `Y`：输出空间

对应的协变混淆记为一个五元组：

- `C = (φX, φΘ, φY, ψY, f~)`
  - `φX`：输入混淆
  - `φΘ`：模型参数变换
  - `φY`：输出混淆
  - `ψY`：输出反混淆
  - `f~`：混淆空间中的推理函数

它满足两个条件：

1. **交换性/近似交换性**
   - 先做明文推理再映射，与先做混淆再在混淆空间推理，结果应近似一致；
   - 误差由 `eC` 控制。

2. **可反混淆**
   - `ψY(φY(y)) = y`

直观理解：

- 普通“数据混淆”是把输入藏起来；
- **协变混淆**是把“输入—模型—输出”这一整条链路，整体搬到一个新的坐标系里执行。

### 3.2 为什么协变混淆优于只混淆数据

论文的理论主张是：

- 在相同精度损失下，协变混淆比数据单边混淆泄露更少。

原因在于：

- 如果只混淆输入，模型仍以明文参数运行，中间表示仍可能保留稳定结构；
- 若连参数也协同变换，攻击者看到的是一个“整体变形过的系统”，更难利用 embedding、attention score、hidden state、head 对齐关系等结构性信息恢复明文。

---

## 4. 论文的方法论框架

### 4.1 组合定理是整个设计的骨架

论文提出三类组合定理，使得可以先为各模块单独设计混淆，再拼成完整模型：

1. **顺序组合**
   - 适用于层与层串联，如线性层接线性层。

2. **并行组合**
   - 适用于注意力里的多路并行计算，如 query/key 分支。

3. **求和组合**
   - 适用于残差连接。

这三条定理的意义非常关键：  
它解释了为什么作者可以分别设计 embedding、attention、FFN、normalization 的混淆，再把它们合并成完整 LLM 的混淆机制。

### 4.2 AloePri 的总流程

论文中的 AloePri 分成两个阶段：

#### 阶段 A：离线模型混淆

客户端本地拿到开源模型后，执行：

1. 生成秘密 token 置换 `τ`；
2. 对 embedding 和 lm head 做 token 级置换；
3. 生成 key matrix / inverse key matrix；
4. 对 attention、FFN、normalization 等权重做协变变换；
5. 得到混淆后的模型并部署到服务端。

#### 阶段 B：在线私有推理

客户端执行：

1. 用秘密词表映射把 prompt 变成“混淆文本”；
2. 把混淆文本发送给服务端；
3. 服务端用混淆模型推理，返回混淆输出；
4. 客户端再做反映射还原结果。

这意味着：

- 服务端**看到的是混淆后的输入、混淆后的模型和混淆后的输出**；
- 而客户端承担的在线开销只有轻量级 token 映射，不参与重计算。

---

## 5. AloePri 具体方法拆解

## 5.1 离线阶段总目标

离线阶段有两个同时要达成的目标：

1. **支持在线 token 级混淆**
   - 这样客户端在线只需做词表映射即可。

2. **不能让服务端从混淆前后权重关系反推出 token 置换**
   - 所以不能只有 embedding/head 的简单行列置换；
   - 还必须在整个模型里引入可抵消的线性变换、噪声、head/block permutation 等结构。

---

## 5.2 Key Matrix 生成：整个方案的“坐标变换器”

论文先设计了一个生成 key matrix 与 inverse key matrix 的算法（Algorithm 1）。

### 5.2.1 输入参数

- `d`：hidden size
- `h`：expansion size
- `λ`：控制矩阵范数的系数

### 5.2.2 核心思路

算法会先生成一组基础矩阵：

- 接近正交的 `B`
- 低秩扩展块 `E`
- 低秩扩展块 `F`
- 大正交矩阵 `Z`

然后分别构造：

- `P^`：key matrix
- `Q^`：inverse key matrix

满足：

- `P^ * Q^ = I`

### 5.2.3 作用

这些矩阵被用于相邻模块之间的双边变换：

- 左乘某个逆变换；
- 右乘某个正变换。

这样做的好处是：

- 前向传播时变换能相互抵消，保持计算正确性；
- 但攻击者若不知道这些矩阵，就难以从权重结构中恢复隐藏的 token permutation。

### 5.2.4 复现时的理解

可以把它理解成：  
作者给每一层状态都引入了一个“隐藏坐标系”，模型虽然还在做近似同样的计算，但所有中间表示都被搬到了一个外部不可见的新坐标系里。

---

## 5.3 Embedding 和 Model Head 混淆

这是 AloePri 最关键的一步，因为输入输出隐私最终都落在 token 级映射上。

### 5.3.1 三种操作同时使用

论文对 embedding / head 使用三类操作：

1. **加噪**
   - `W_embed* = W_embed + αe * E_embed`
   - `W_head* = W_head + αh * E_head`

2. **token 级置换**
   - 采样秘密置换 `τ ~ S_n`
   - 构造对应置换矩阵 `Π`

3. **key matrix 变换**
   - embedding 右乘 `P^embed`
   - head 左乘 `Q^head`

### 5.3.2 最终形式

论文给出的形式为：

- `W~embed = Π W_embed* P^embed`
- `W~head = Q^head W_head* Π^T`

### 5.3.3 为什么这样设计

这样一来：

- 输入 token 被 `τ` 改写后，仍能在置换后的 embedding 行上取到对应向量；
- 输出 logits 在 head 上也会被同一个 `τ` 对应地置换；
- 自回归生成仍然可行，因为整个词表空间是做了一个一一映射，而不是删词或改维度。

### 5.3.4 词表映射的角色

客户端会用 `τ` 构造秘密词表映射 `Z`，在线时把原 token 映射到混淆 token，再在输出端反向映射回来。

这一步是论文能兼容生成任务的关键：

- 它不是把 embedding 当通信载荷发给服务器；
- 它发的仍然是“文本/词元序列”，因此更接近现有 LMaaS 接口。

---

## 5.4 Attention 混淆

Attention 是整篇论文最复杂、也是最有技术含量的部分。

作者把 attention 混淆分成两层：

1. **head 内变换（intra-head transformation）**
2. **head 间置换（inter-head permutation）**

### 5.4.1 Head 内变换

对一组 attention head 的 `Wq, Wk, Wv, Wo`：

- 使用 `Q^q, Q^k, Q^v, P^o` 做线性变换；
- 对 `value/output` 使用额外可逆矩阵 `U^vo`；
- 对 `query/key` 再引入：
  - 2x2 rotary matrix `R^qk`
  - scaling matrix `H^qk`
  - block permutation `Z^block`

作者的动机是：

- RoPE 使 query/key 带有严格的位置块结构；
- 只做一般线性变换可能破坏 RoPE 对应关系；
- 因此需要围绕 RoPE 的 2x2 block 设计特殊变换。

### 5.4.2 Block permutation

论文发现：

- 在有限窗口内对 RoPE 的 2x2 block 做置换，对精度影响较小；
- 尤其是较大 index 的 block，更容易在不伤精度的前提下置换。

所以作者又设计了一个 block permutation 采样算法（Algorithm 2），参数包括：

- `β`：最大窗口大小
- `γ`：窗口采样参数
- `ζ`：RoPE 相关参数
- `m_blocks`：block 数量

### 5.4.3 Head 间置换

作者还会采样：

- `τkv`
- `τgroup`

用于：

- 在 key/value head 级别重排；
- 在 grouped-query 结构内重排 query/output head。

这样做的目的不是算子正确性，而是**打断 head 之间的统计相关性**，防止攻击者利用 head 对齐关系做恢复。

### 5.4.4 兼容的 attention 结构

论文明确声称该设计可适用于：

- MHA
- MQA
- GQA
- MLA

其中 MLA 需要对低秩 query/key 部分额外做一套可逆变换。

### 5.4.5 方法理解

attention 部分的保护其实针对两类信息：

1. **value path / hidden state**
   - 通过 key matrix 等线性变换隐藏；

2. **attention score**
   - 仅靠 key matrix 不够；
   - 还要靠 head permutation 和 RoPE block permutation 进一步隐藏。

这也是论文后续消融实验的核心结论之一。

---

## 5.5 FFN 混淆

FFN 的主要难点在于存在非线性 `SiLU` 和门控结构，所以不能随意做一般矩阵变换。

论文选择的思路是：

- 主要用 **置换 + 缩放**，而不是任意线性变换。

对 dense FFN：

- `W~gate = Q^gate W_gate Z^ffn`
- `W~up = Q^up W_up H^ffn Z^ffn`
- `W~down = (Z^ffn)^-1 (H^ffn)^-1 W_down P^down`

其中：

- `Z^ffn` 是随机置换矩阵；
- `H^ffn` 是随机缩放矩阵。

### 5.5.1 为什么只用置换和缩放

因为 FFN 中的核心是：

- `SiLU(xW_gate) ⊙ (xW_up)`

若做一般线性变换，容易破坏逐元素非线性和 Hadamard 乘法结构。  
置换与缩放更容易在输出端精确抵消，从而保持函数结构不变。

### 5.5.2 MoE 的额外处理

对 MoE，论文还处理了 router：

- 先对 `W_router` 做 expert 维度的归一化；
- 再用 `Z^router` 对 expert 顺序置换；
- 同时做相应线性变换。

目标是：

- 专家选择逻辑不变；
- 但攻击者看不到真实 expert 对应关系。

---

## 5.6 Layer Normalization 变换

论文主要讨论 RMSNorm。

### 5.6.1 问题

如果输入状态已经被 `P^` 变换到新空间，直接做 RMSNorm 会改变范数统计，导致归一化结果偏移。

### 5.6.2 论文做法

作者定义一个系数：

- `κ = E[ ||xP^|| / ||x|| ]`

用于校正因 `P^` 引入的偏差。

随后把明文 RMSNorm 替换为：

- 一个系数修正后的 RMSNorm；
- 再融合一个线性层到相邻层中。

### 5.6.3 作用

它本质上是在解决：

- 线性可逆变换后的归一化统计偏移问题。

这一步不如 attention/embedding 显眼，但如果不做，整网误差会不断积累。

---

## 5.7 在线推理流程

在线阶段论文的设计非常“工业友好”：

1. 客户端本地 tokenize；
2. 用秘密词表映射把 token 换成混淆 token；
3. 再 detokenize 成混淆文本，发送给服务端；
4. 服务端按正常接口重新 tokenize 后，用混淆模型推理；
5. 返回混淆输出；
6. 客户端用词表映射逆操作解码。

### 5.7.1 工程上的重要意义

它没有要求：

- 客户端参与分层推理；
- 客户端发送 hidden states；
- 服务端修改 LMaaS API 为 embedding 级接口。

所以它更容易嫁接到现有推理服务。

---

## 6. 论文的正确性与安全性分析

## 6.1 精度分析

论文对 embedding/head、attention、FFN、LayerNorm 分别分析混淆误差，再借助顺序/并行/残差组合定理把误差累积到整网。

核心思想是：

- 每个模块都构造一个近似满足交换性的 covariant obfuscation；
- 每个模块的误差可控；
- 整体误差由模块误差和 Lipschitz 常数共同上界。

因此，AloePri 的精度损失来自：

- embedding/head 加噪；
- attention block permutation；
- LayerNorm 近似校正；
- 数值精度误差（尤其 BF16 下更明显）。

## 6.2 安全性分析

作者把隐私核心归结为：  
攻击者能否恢复秘密 token permutation `τ`。

### 6.2.1 信息泄露分解

作者把总泄露分成两部分：

1. **静态泄露**
   - 来自离线阶段暴露给服务端的混淆模型权重；

2. **动态泄露**
   - 来自在线阶段反复观察到的混淆输入/输出 token 频率。

论文给出一个 mutual information 上界，说明总泄露不超过这两者之和。

### 6.2.2 静态攻击成功率上界

论文进一步基于 PAC Privacy 框架给出静态攻击成功率上界。  
从工程角度可以提炼出三点：

- embedding/head 噪声越足，越难从权重反推 permutation；
- 变换群越丰富，越不容易被结构恢复；
- 但噪声和变换过强会伤精度、伤数值稳定性。

### 6.2.3 论文的安全目标边界

论文明确强调：  
它追求的是 **面向受限攻击者的实用安全性**，而不是理想世界下的密码学最强安全。

这点非常重要，复现时不要误读成：

- “它等价于 FHE/安全多方计算”；
- “它可在任意攻击模型下严格隐藏一切信息”。

它更准确的定位是：

- 在工业部署约束下，以极小在线代价换取显著隐私增强。

---

## 7. 论文评估了哪些攻击

论文实验覆盖三类攻击：

### 7.1 混淆恢复类

- **VMA（Vocabulary-Matching Attack）**
  - 从已知明文/混淆权重的结构关系里恢复 permutation；
  - 针对 embedding、head、query/key、FFN、router 等多组权重组合做匹配。

- **IA（Invariant Attack）**
  - 利用变换下不变统计量做恢复；
  - 论文具体分析了：
    - Gate-IA
    - Attn-IA

### 7.2 训练型反演类

- **NN attack**
- **IMA（Inversion Model Attack）**
  - 训练一个模型把混淆 embedding / hidden state 反推出明文 embedding/token。

- **ISA（Internal State Attack）**
  - 利用隐藏层状态或 attention score 反推输入。

### 7.3 Token 频率利用类

- **TFMA**
  - 基于 token 频率做直接匹配。

- **SDA**
  - 基于替换密码建模和 Transformer 解码恢复文本。

这组攻击覆盖得比较全面，因此复现时也应尽量按这三类结构搭建评测。

---

## 8. 实验结果怎么理解

## 8.1 与基线比较

论文在 Qwen2.5-14B-Instruct 上和 SANTEXT、RANTEXT、DP-Forward、SGT 比较：

- AloePri 的精度损失远小于其他混淆方法；
- 同时对 VMA、IMA、IA、ISA 等攻击更稳健；
- 作者还用 CLUB 估计 mutual information，声称 AloePri 泄露最低。

## 8.2 默认隐私超参数

论文默认推荐参数大致为：

- `αe = 1.0`
- `αh = 0.2`
- `λ = 0.3`
- `h = 128`
- `β = 8`
- `γ = 1e3`

这些参数是后续小规模复现的首选起点。

## 8.3 参数敏感性结论

### 噪声参数

- `αe`、`αh` 太小：
  - VMA 更容易恢复 token；
  - 原因是权重关系过于接近“纯置换”。

### key matrix 系数 `λ`

- `λ` 太大时，尤其在 BF16 下容易数值溢出；
- 论文给出经验结论：`λ = 3.0` 在 BF16 下会显著伤精度。

这说明复现时必须关注：

- 数值稳定性；
- dtype；
- 中间状态范数分布。

## 8.4 内部状态保护消融

论文的结论是：

- 只有 embedding noise：不够；
- noise + key matrix：可以明显保护 hidden state；
- 但 attention score 仍脆弱；
- 还要加 head/block permutation，attention score 才能真正被保护。

这条结论直接决定了复现优先级：

- attention 相关混淆不是可选优化，而是核心组件。

## 8.5 Token 频率泄露结果

论文承认 deterministic token permutation 天然保留频率统计。  
但在实验中：

- TFMA 的 Top-100 token 恢复率仍较低；
- SDA 的 BLEU-4 也很低；
- 说明频率泄露存在，但不足以恢复有意义文本。

## 8.6 效率结果

论文报告：

- 离线混淆对 32B 级模型约 10 分钟；
- 对 671B 模型约 8 小时；
- 在线 TTFT / TPOT 与明文推理近似等价。

对复现而言，这意味着：

- **真正昂贵的是离线模型预处理**；
- **在线阶段几乎不应显著慢于原生推理**。

---

## 9. 复现这篇论文时，应该怎样拆目标

我建议把复现分成三个层级，不要一开始就追求“全量工业系统复现”。

## 9.1 Level 1：算法级复现

目标：

- 在 `transformers` 上实现 AloePri 的核心数学机制；
- 不要求 vLLM 集成；
- 不要求大模型和大规模 benchmark。

交付物：

- key matrix 生成；
- embedding/head 混淆；
- attention 混淆；
- FFN 混淆；
- LayerNorm 校正；
- 在线 token 映射推理链路。

验证标准：

- 混淆前后输出在小样本上可用；
- 输入/输出映射链路正确；
- 模型能完成自回归生成。

## 9.2 Level 2：实验级复现

目标：

- 在中小模型上复现实验趋势，而不是完全复现论文全部数值。

建议模型：

- 优先从你当前已有的 `Qwen2.5-0.5B-Instruct` 开始；
- 再升级到 1.5B / 7B / 14B。

建议验证：

- 精度：简单 instruction-following、分类、常识问答；
- 隐私：先复现 VMA、NN/IMA，再逐步补 ISA、TFMA、SDA。

验证标准：

- 随 `αe / αh / λ` 调参能观察到论文同方向趋势；
- 加入 attention block/head permutation 后，内部状态恢复显著变难。

## 9.3 Level 3：系统级复现

目标：

- 把混淆模型接入高性能推理框架；
- 观察 TTFT / TPOT 是否接近原生模型。

建议顺序：

1. 先 `transformers` 单卡；
2. 再尝试 vLLM；
3. 最后再考虑多卡和更大模型。

---

## 10. 面向当前仓库的具体复现方案

下面这部分是**我根据论文内容给出的工程化落地建议**，属于复现方案，而不是论文原文内容。

## 10.1 复现目标定义

第一阶段先实现一个 **最小可运行 AloePri 原型**：

- 模型：`Qwen2.5-0.5B-Instruct`
- 框架：`transformers`
- 设备：CPU 或单卡 GPU
- 任务：单轮文本生成
- 目标：证明“混淆模型 + 混淆 token 输入 + 反混淆输出”全流程能跑通

## 10.2 建议的代码结构

建议后续在仓库中增加如下模块：

- `aloepri/key_matrix.py`
  - 实现 Algorithm 1

- `aloepri/token_mapping.py`
  - 生成 `τ`、`Π`、秘密词表映射 `Z`

- `aloepri/embed_head.py`
  - embedding/head 的噪声、置换与双边变换

- `aloepri/attention.py`
  - attention head 内变换、head/block permutation

- `aloepri/ffn.py`
  - dense FFN 与 MoE router 变换

- `aloepri/layernorm.py`
  - RMSNorm 校正与融合

- `aloepri/obfuscate_model.py`
  - 遍历 Hugging Face 模型权重并输出混淆模型

- `aloepri/inference.py`
  - 在线 token 混淆、服务端推理、输出反混淆

- `scripts/run_obfuscated_infer.py`
  - 命令行测试入口

- `scripts/eval_vma.py`
  - 静态词表恢复攻击

- `scripts/eval_privacy_accuracy.py`
  - 精度/隐私联合评估

## 10.3 第一阶段实施步骤

### Step 1：先做 token permutation + embed/head 混淆

这是最小闭环：

1. 对 tokenizer 构造 `τ`；
2. 重排 embedding/head；
3. 加 embedding/head 噪声；
4. 在线把 token 映射成混淆 token；
5. 跑生成并反映射输出。

这一阶段先不改 attention / FFN，只验证：

- 词表映射逻辑是否闭环；
- 自回归生成是否仍成立。

### Step 2：接入 key matrix，打通隐层坐标变换

接着实现 Algorithm 1，把相邻层之间的表示空间变换接起来。

重点检查：

- 维度是否一致；
- `P^ Q^ = I` 是否数值上成立；
- BF16/FP16 下是否溢出；
- 与 residual 相接时是否正确抵消。

### Step 3：实现 FFN 混淆

FFN 相对 attention 更容易先做，因为结构比较规则。  
先做 dense FFN：

- `gate/up/down` 的置换与缩放；
- 验证前向输出偏差。

若后续做 MoE，再补 router。

### Step 4：实现 attention 混淆

建议拆成四步：

1. 只做 `Q/K/V/O` 的基本双边线性变换；
2. 加 `U^vo`；
3. 加 `R^qk + H^qk`；
4. 最后加 block permutation 与 head permutation。

原因是 attention 最容易出 bug，必须逐步加复杂度。

### Step 5：实现 RMSNorm 校正

这一步建议在 attention/FFN 都基本正确后再接入。  
否则很难判断误差来自哪一块。

### Step 6：做小规模评测

最先做三类检查：

1. **功能正确性**
   - 是否能生成；
   - 反混淆文本是否可读。

2. **精度保持**
   - 若干固定 prompt 上，与原模型输出做相似度比较；
   - 再做小型 benchmark。

3. **隐私增强**
   - 先做最容易复现的 VMA / NN / IMA；
   - 再补 ISA / TFMA / SDA。

---

## 11. 复现时最需要重点盯住的技术难点

## 11.1 Tokenizer 与词表映射的一致性

这是最容易被低估的问题：

- token 置换必须和 tokenizer 的编码/解码行为完全一致；
- 若 BPE merge 规则在 detokenize/re-tokenize 中引入差异，在线映射可能失效。

建议：

- 先在 token id 层做映射，不要先从字符串层做复杂替换；
- 为 encode/decode 循环写严格单测。

## 11.2 Attention 的 RoPE 块结构

论文对 block permutation 的设计是围绕 RoPE 的 2x2 block 来做的。  
如果实现时忽略这一点，很可能：

- 精度急剧下降；
- 或 attention score 结构被破坏。

## 11.3 数值稳定性

必须持续监控：

- hidden state norm；
- logits 范围；
- BF16/FP16 overflow；
- `λ`、`h` 增大后的爆炸风险。

建议初期全部用：

- `float32`

等逻辑跑通后再回到：

- `bfloat16`

## 11.4 内部状态保护不能只靠加噪

论文消融已经说明：

- noise 只能解决一部分问题；
- 真正让 attention score 也变安全，必须引入 head/block permutation。

所以如果你的初版只做噪声和 embed/head 置换，很可能：

- 看起来能跑；
- 但隐私性并不等价于论文方案。

## 11.5 论文强调的是“趋势复现”，不是“数值逐点一致”

由于你当前环境和论文不同：

- 模型规模不同；
- 硬件不同；
- 推理框架不同；
- 可能没有官方代码；

更合理的目标是复现以下结论趋势：

- 适中噪声下精度损失较小；
- 噪声不足时 VMA 更容易恢复；
- `λ` 过大时数值稳定性变差；
- attention permutation 能显著提升对 ISA 的防御；
- 在线延迟增长远小于离线混淆成本。

---

## 12. 我建议的复现路线图

### 第 1 周：最小原型

- 跑通 `Qwen2.5-0.5B-Instruct` 明文推理；
- 实现 token permutation；
- 实现 embedding/head 混淆；
- 做闭环生成测试。

### 第 2 周：完整权重混淆

- 实现 key matrix；
- 接入 FFN；
- 接入 attention；
- 接入 RMSNorm 校正。

### 第 3 周：评测脚本

- 小规模 accuracy benchmark；
- VMA / NN / IMA 评测；
- 参数敏感性扫描：`αe`、`αh`、`λ`。

### 第 4 周：系统扩展

- 尝试更大模型；
- 尝试 `vLLM`；
- 记录 TTFT / TPOT。

---

## 13. 对你当前复现任务的直接建议

结合你当前仓库状态，我建议下一步按下面顺序推进：

1. 先基于当前已下载的 `Qwen2.5-0.5B-Instruct`，做一个 **AloePri 最小可跑版本**；
2. 第一版先只实现：
   - token permutation
   - embedding/head noise
   - embedding/head permutation
   - 在线反混淆
3. 跑通后再补：
   - key matrix
   - FFN
   - attention
   - RMSNorm
4. 最后再做攻击评测和参数扫描。

这是因为：

- embedding/head + token mapping 是 AloePri 的最小主链路；
- 一上来全做 attention/FFN/RMSNorm，调试成本会很高；
- 小模型上先复现结构正确性，比直接追论文完整数值更现实。

---

## 14. 结论

从论文内容来看，AloePri 的真正创新点不是“加噪”或“置换”本身，而是：

- 把 **token permutation、模型参数协同变换、attention 结构保护、FFN 可抵消变换、RMSNorm 校正** 组合成一个完整的 **covariant obfuscation** 体系；
- 让 LLM 能在混淆空间中保持近似原有推理能力；
- 同时保留工业系统最关心的在线效率和基础设施兼容性。

如果要复现，最重要的是不要把它误简化成“embedding 加噪声”方案。  
真正的 AloePri 至少包含以下四个不可或缺的层面：

- token 级秘密映射；
- 全模型权重协同变换；
- attention 特化保护；
- 误差与数值稳定性的联合控制。

---

## 15. 推荐的下一步

如果继续推进，我建议下一步直接在当前仓库进入实现阶段，先做以下两件事：

1. 写一个 `AloePri` 最小原型，先支持 `Qwen2.5-0.5B-Instruct`；
2. 用 `transformers` 打通：
   - token 混淆
   - embedding/head 混淆
   - 反混淆生成

等这个最小闭环跑通后，再逐步把 attention、FFN、RMSNorm 和评测脚本补上。

