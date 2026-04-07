# 阶段 E 复杂 Attention 结构：排错与修正报告

## 1. 报告目的

本文档记录阶段 E 从“复杂 attention 结构接入后显著退化”到“问题被精确定位并完成修复”的全过程。

这份报告重点回答四个问题：

1. **问题最开始表现成什么样**
2. **问题是怎么被一步步收敛定位的**
3. **最终到底改了哪些代码**
4. **修复后系统恢复到了什么状态**

这份报告与 `docs/阶段E复杂Attention复现报告.md` 的区别是：

- `阶段E复杂Attention复现报告.md` 侧重当前阶段 E 的结构与实验现状；
- 本文档侧重**排错过程、修复过程、修复结果**。

---

## 2. 修复前的阶段 E 问题概述

在最初的阶段 E 实验里，我们引入了论文第 `5.2.3` 节中的复杂 attention 结构：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`
- `τ_kv`
- `τ_group`

但最早的 block0 单层消融结果显示，系统相较阶段 D 的 simplified attention 基线明显退化。

### 2.1 修复前 block0 关键结果

在第一次阶段 E 单层消融中，观测到：

| Profile | `attn_out` max error | `block_out` max error | `final_logits` max error |
|---|---:|---:|---:|
| `simplified` | `0.000529` | `0.024588` | `35.3178` |
| `rqk` | `0.295059` | `2.246665` | `36.9155` |
| `rqk_hqk` | `0.296669` | `2.141420` | `37.0151` |
| `rqk_hqk_block` | `0.105283` | `1.772131` | `35.4460` |

初始结论是：

- 一旦引入 `R̂_qk`，系统立刻明显劣化；
- `Ĥ_qk` 没有显著把误差拉回去；
- `Ẑ_block` 只能部分缓解；
- `τ_kv / τ_group` 对恢复指标几乎没有可见影响。

### 2.2 修复前 prefix 多层结果

在修复前的 prefix 实验中：

- `prefix_layers_2`
  - `layer_0_block_out ≈ 1.7721`
  - `layer_1_block_out ≈ 2.1576`
  - greedy 首 token 一致率：`0.0`

- `prefix_layers_4`
  - `layer_2_block_out ≈ 456.63`
  - `layer_3_block_out ≈ 1282.63`

这意味着：

> 修复前的复杂 attention 结构一旦推广到多层，就会非常快地破坏当前阶段 D 已经建立的稳定传播链。

---

## 3. 排错思路总览

这次排错不是“随便试参数”，而是遵循一个明确的理论定位链路。

### 3.1 已有事实

阶段 A ～ D 已经证明：

- 词表空间闭环正确；
- hidden-space 简化 key 正确；
- simplified attention 正确；
- RMSNorm `κ` 修正正确；
- FFN wrapper 正确；
- 多层 block 级恢复正确。

因此，当阶段 E 出现退化时，最合理的判断不是：

- “整个系统都可能有问题”

而是：

> **问题最可能集中在复杂 attention 新增结构本身。**

### 3.2 第一轮定位的核心假设

最初有三类高概率嫌疑：

1. `RMSNorm` / `residual` / `FFN` 链路被复杂 attention 牵连；
2. 复杂 attention 自身没有满足论文要求的严格配对结构；
3. GQA 头级 permutation (`τ_kv / τ_group`) 在当前实现里没有真正生效。

### 3.3 排错原则

本次排错采取了三条严格原则：

1. **先验证理论硬约束是否成立**
   - 比如 `A · B^T = I`
2. **先定位误差从哪个中间量开始出现**
   - 不能只看最终 logits
3. **先修最核心结构性 bug，再看多层传播**
   - 不先去调更深层和更多 profile

---

## 4. 第一轮精确定位：attention score 本身已经坏了

为了区分“是 value/output 链路坏了”还是“Q/K score 本身就坏了”，做了一个非常关键的精确诊断：

### 4.1 诊断方法

在 block0 上直接取出：

- `q_proj(hidden)`
- `k_proj(hidden)`

然后比较：

\[
QK^\top
\]

与引入复杂结构后的：

\[
(Q A)(K B)^\top
\]

在经过 Qwen2 的 `apply_rotary_pos_emb(...)` 之后，attention score 是否仍与 baseline 一致。

这一步只看 attention score，不掺入：

- `v_proj`
- `o_proj`
- residual
- FFN
- RMSNorm

### 4.2 预期

若 `R̂_qk / Ĥ_qk` 的设计与 RoPE 兼容，那么至少在：

- `rqk`
- `rqk_hqk`

这些 profile 下，attention score 应该近似不变。

### 4.3 修复前实际结果

修复前测到：

- `rqk` score max error：`412.2076`
- `rqk_hqk` score max error：`408.5590`
- `rqk_hqk_block` score max error：`337.1216`

这说明：

> 问题不是出在后面的 `V/O` 或 residual，而是 **Q/K score 在引入复杂结构后已经直接坏掉了**。

这是整个排错过程里最关键的一步定位。

---

## 5. 第一轮根因定位：RoPE block 组织方式和 Qwen 实现不匹配

在看到 score 级错误后，继续检查 Qwen2 的 RoPE 具体实现。

### 5.1 检查到的关键事实

Qwen2 的 `rotate_half(x)` 不是按相邻的 `2x2` block 组织，而是：

- 取前半维 `x1`
- 取后半维 `x2`
- 拼成 `(-x2, x1)`

也就是说，Qwen2 的 RoPE 配对方式是：

\[
(i,\ i + d/2)
\]

而不是：

\[
(2i,\ 2i+1)
\]

### 5.2 当前实现的错误

在修复前，`src/attention_keys.py` 中生成：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`

的逻辑，默认按**相邻维度组成 2x2 block**。

这与 Qwen2 的 `rotate_half` 语义不一致。

### 5.3 这为什么会导致大错

虽然从论文抽象看，我们要求：

\[
A B^\top = I
\]

当前代码也满足了这个矩阵恒等式；

但如果 `A` 和 `B` 的 block 组织方式与模型实际的 RoPE 维度配对不一致，那么：

- 形式上你在做“二维旋转”
- 实际上你旋转的不是模型 RoPE 认为的一对维度

结果就是：

> attention score 在进入 softmax 之前就已经偏离。

### 5.4 第一轮修复方案

对 `src/attention_keys.py` 做了结构性修复：

- `generate_r_qk(...)`
  - 改成按 `(i, i + head_dim/2)` 成对生成二维旋转；

- `generate_h_qk(...)`
  - 改成前半和后半维共享同一个 scale；

- `generate_block_perm(...)`
  - 改成以这种“前半 / 后半配对”作为 block 单元。

### 5.5 第一轮修复后的理论检查

修复后重新测得：

- `rqk` score max error：`0.0048828125`
- `rqk_hqk` score max error：`0.0048828125`
- `rqk_hqk_block` score max error：`0.0048828125`

并且：

- `A · B^T - I` 的最大误差仍为 `1.19e-07`

这说明：

> 第一轮修复已经把最致命的 Q/K score 破坏问题修掉了。

### 5.6 对应证据文件

修复后的 score 诊断结果已保存到：

- `outputs/stage_e/diagnostics_after_fix.json`

---

## 6. 第二轮定位：q/k/v 指标没有被显式汇总

第一轮修复后，attention score 理论问题解决了，但还需要把更多中间结果显式导出。

### 6.1 发现的问题

虽然 `ComplexQwen2Attention` 内部已经记录了：

- `layer_i_q_proj_out`
- `layer_i_k_proj_out`
- `layer_i_v_proj_out`

但原有摘要逻辑没有把这些指标写入结果文件。

结果就是：

- `stage_e` 的 summary 中
  - `q_proj_out`
  - `k_proj_out`
  - `v_proj_out`
  
这些项是 `None`。

### 6.2 第二轮修复方案

在 `src/stage_d.py` 的汇总逻辑中，把：

- `q_proj_out`
- `k_proj_out`
- `v_proj_out`

加入到显式摘要里。

同时注意：

- 这些张量不是 hidden-size 维，不能按 hidden transform 去做 inverse restore；
- 对它们应直接与 baseline 对应量做比较。

### 6.3 修复后的 block0 关键中间量

修复后 `outputs/stage_e/block0_attention_complex.json` 中已经能看到：

#### `simplified`

- `q_proj_out max error = 0.019803`
- `k_proj_out max error = 0.010513`
- `v_proj_out max error = 0.000426`

#### `rqk`

- `q_proj_out max error = 0.019803`
- `k_proj_out max error = 0.010511`
- `v_proj_out max error = 0.000426`

#### `rqk_hqk_block_taukv_taugroup`

- `q_proj_out max error = 0.019803`
- `k_proj_out max error = 0.010513`
- `v_proj_out max error = 0.000426`

这说明：

> 经过第一轮修复后，复杂 attention 的 Q/K/V 线性输出已经重新与 simplified 版本对齐到了数值噪声级别。

---

## 7. 第三轮定位：`Ẑ_block` 当前实现过于激进

虽然第一轮修复已经让 `rqk / rqk_hqk` 恢复到和 simplified 几乎一致，但仍需检查 `Ẑ_block`。

### 7.1 修复前的情况

最初的 `Ẑ_block` 采用的是一种较粗糙的：

- 固定窗口
- 窗口内均匀随机置换

这会导致早期低频 RoPE blocks 也被过度打乱。

### 7.2 排查思路

论文原意并不是“把所有 block 在窗口内等概率乱排”，而是：

- 以 block 为单位；
- 以局部窗口为限；
- 更偏向对高 index block 进行更积极的扰动；
- 低 index block 更应保守。

### 7.3 第三轮修复方案

在 `src/attention_keys.py` 中把 `generate_block_perm(...)` 升级为：

- 支持 `dynamic_window`
- 根据 `rope_base` 和 `gamma` 生成更接近论文的窗口长度采样
- 保证低 index block 更不容易发生扰动

### 7.4 修复后的 block0 结果

修复后 block0 结果变成：

#### `rqk_hqk_block`

- `attn_out max error = 0.000529`
- `block_out max error = 0.024588`

与 `simplified` 基本一致。

#### `rqk_hqk_block_taukv`

- 同样与 `simplified` 基本一致。

#### `rqk_hqk_block_taukv_taugroup`

- 同样与 `simplified` 基本一致。

这说明：

> 第二轮和第三轮修复后，复杂 attention 结构已经不再破坏 block0 的恢复质量。

---

## 8. 修复后的 block0 单层消融结果

当前修复后的 block0 单层消融结果如下：

| Profile | `q_proj_out` | `k_proj_out` | `v_proj_out` | `attn_out` | `block_out` | `final_logits` |
|---|---:|---:|---:|---:|---:|---:|
| `simplified` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk` | 0.019803 | 0.010511 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk_block` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk_block_taukv` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk_block_taukv_taugroup` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |

### 结论

当前可以明确说：

> block0 单层上，复杂 attention 结构版已经恢复到与阶段 D simplified attention 基线数值一致。

这意味着：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`
- `τ_kv`
- `τ_group`

至少在**功能正确性**层面，不再破坏当前恢复链路。

---

## 9. 修复后的 prefix 多层结果

## 9.1 prefix-2

来自：

- `outputs/stage_e/prefix_layers_2.json`

关键结果：

- `layer_0_block_out = 0.024588`
- `layer_1_block_out = 0.030693`
- `final_logits = 31.2628`

这与阶段 D 的 `layers_2` 基线已经一致。

## 9.2 prefix-4

来自：

- `outputs/stage_e/prefix_layers_4.json`

关键结果：

- `layer_0_block_out = 0.024588`
- `layer_1_block_out = 0.030693`
- `layer_2_block_out = 3.182129`
- `layer_3_block_out = 29.994019`
- `final_logits = 31.7846`

这同样和阶段 D 的 `layers_4` 结果一致。

## 9.3 prefix-8

来自：

- `outputs/stage_e/prefix_layers_8.json`

关键结果：

- `layer_7_block_out = 30.861206`
- `final_logits = 31.914819`

依旧与阶段 D 的 `layers_8` 基线一致。

## 9.4 prefix-full（24 层）

来自：

- `outputs/stage_e/prefix_layers_full.json`

关键结果：

- `layer_23_block_out = 7.975414`
- `avg_final_logits_restored_max_abs_error = 20.0770`
- `greedy_first_token_match_rate = 0.8`
- `generated_ids_exact_match_rate = 0.4`
- `generated_text_exact_match_rate = 0.4`

逐 prompt 现象：

- Prompt 4 和 Prompt 5 完全一致；
- Prompt 2 和 Prompt 3 首 token 一致，但全文未完全一致；
- Prompt 1 能恢复出语义正确的自然语言句子。

这和阶段 D 的全层结果对齐，说明：

> 修复后的复杂 attention 版本已经重新回到阶段 D 的系统级稳定链路上。

---

## 10. 关于 `τ_kv / τ_group` 的最终判断

修复后我们仍然看到一个事实：

- `rqk_hqk_block`
- `rqk_hqk_block_taukv`
- `rqk_hqk_block_taukv_taugroup`

在**恢复到 base basis 后的功能指标**上是完全一致的。

这需要正确解读。

### 不应该直接得出的错误结论

不能直接说：

> `τ_kv / τ_group` 没有效果，所以实现肯定错了

因为当前我们衡量的是：

- restored `q/k/v`
- restored `attn_out`
- restored `block_out`
- restored `final_logits`

这些本来就应当在论文设计下尽量保持不变。

### 第四轮定位：显式导出 raw head-level trace

为彻底回答这个问题，我们新增了：

- `scripts/run_stage_e_head_trace_check.py`

并在 `ComplexQwen2Attention` 内部显式记录了未恢复前的原始 head-level 中间量：

- `q_heads_pre_inter_raw`
- `k_heads_pre_inter_raw`
- `v_heads_pre_inter_raw`
- `q_heads_post_inter_raw`
- `k_heads_post_inter_raw`
- `v_heads_post_inter_raw`
- `attn_heads_pre_inverse_raw`

输出结果保存在：

- `outputs/stage_e/head_trace_check.json`

### 第四轮诊断结果

#### 配置一：`rqk_hqk_block`

- `tau_kv = None`
- `tau_group = None`

#### 配置二：`rqk_hqk_block_taukv`

- `tau_kv = [1, 0]`
- `tau_group = None`

#### 配置三：`rqk_hqk_block_taukv_taugroup`

- `tau_kv = [1, 0]`
- `tau_group = [2, 4, 6, 0, 3, 1, 5]`

并且对 raw trace 做 profile-to-profile 比较后得到：

##### `rqk_hqk_block` vs `rqk_hqk_block_taukv`

- `q_heads_post_inter_raw = 55.9384`
- `k_heads_post_inter_raw = 272.7900`
- `v_heads_post_inter_raw = 0.3084`
- `attn_heads_pre_inverse_raw = 0.3084`

##### `rqk_hqk_block_taukv` vs `rqk_hqk_block_taukv_taugroup`

- `q_heads_post_inter_raw = 89.9669`
- `attn_heads_pre_inverse_raw = 0.4094`

### 第四轮修正：避免 `τ_kv / τ_group` 采样到恒等排列

在做这轮诊断时，又发现了一个容易误导排查的工程问题：

- 对 `num_kv_heads = 2` 这种小 head 数量，随机采样 `τ_kv` 时有较高概率得到恒等排列；
- 一旦种子碰巧给出恒等排列，你就会误判“`τ_kv` 没有作用”。

因此，我们对：

- `generate_tau_kv(...)`
- `generate_tau_group(...)`

做了修正：

- 若采样到恒等排列，则重采样若干次；
- 若仍是恒等排列，则退化为简单循环移位，保证实验里真正出现非平凡 permutation。

### 最终解释

因此，现在可以明确写出：

> `τ_kv / τ_group` 在恢复后的功能指标上保持中性，但在未恢复前的 raw head-level trace 上已经被证明确实生效。

这意味着：

- 它们不是 no-op；
- 也不是当前功能恢复链路中的 bug；
- 它们现在扮演的是“改变内部排列、但不破坏最终恢复”的角色。

---

## 11. 本次排错中真正解决的问题

这次多轮排错，最终真正解决的是一个**精确且关键的结构 bug**：

### 问题定义

> 复杂 attention 中 `R̂_qk / Ĥ_qk / Ẑ_block` 的维度配对方式，与 Qwen2 的 `rotate_half` 语义不一致。

### 具体表现

- 修复前：
  - `rqk` score max error ≈ `412`
  - `rqk_hqk` score max error ≈ `409`
  - `rqk_hqk_block` score max error ≈ `337`

- 修复后：
  - 三者 score max error 都降到 `0.0048828125`

### 本质

修复前我们把“论文的 2D block”错误地实现成了：

- 相邻维配对

而 Qwen2 的 RoPE 实际上是：

- 前半维 / 后半维配对

修复后把：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`

全部改成和这个真实配对方式一致，于是功能恢复链重新成立。

---

## 12. 本次排错没有发现的问题

这次多轮排错同时也说明，有一些原先怀疑的点并不是当前主因：

### 不是主因的 1：RMSNorm

阶段 C/D 已经证明：

- `κ + permuted norm weight` 的链路是成立的。

阶段 E 修复后也说明：

- 复杂 attention 不再破坏 norm 链路。

### 不是主因的 2：Residual

如果 residual basis 不一致，修复 attention 后依然不可能恢复到阶段 D 水平。

但现在阶段 E 已经重新回到了阶段 D 的多层结果，因此 residual 不是当前主因。

### 不是主因的 3：FFN

FFN wrapper 在阶段 C/D 已经证明工作正常。  
阶段 E 修复后 prefix 多层与阶段 D 对齐，也说明 FFN 不是当前这轮 bug 的主来源。

---

## 13. 当前阶段 E 的正式结论

基于本次多轮排错与修复，可以给出如下正式结论：

### 结论 1

阶段 E 中导致系统显著退化的主因，已经被**精确定位并修复**：

> **复杂 attention 的 Q/K 结构矩阵与 Qwen2 的实际 RoPE pairing 不一致。**

### 结论 2

修复后：

- block0 单层复杂 attention 结构版已经恢复到与 simplified baseline 一致；
- prefix-2 / 4 / 8 / full 也重新与阶段 D 的多层稳定链路对齐。

### 结论 3

`τ_kv / τ_group` 的内部作用已经被显式验证，且不再存在“是否真正生效”的不确定性。

### 结论 4

当前复杂 attention 路径已经不再是系统稳定性的主要阻碍。

因此，阶段 E 现在可以从原来的：

> “结构接入完成，但结果待排查”

升级为：

> **“复杂 attention 结构接入完成，并已完成关键功能性修复。”**

---

## 14. 后续建议

虽然阶段 E 的主功能 bug 已经修复，但还有两个可以继续做的增强项：

### 建议 1：继续细化 raw head-level / attention-map trace

当前已经验证了：

- `τ_kv / τ_group` 会改变恢复前的 head-level raw trace。

下一步如果要继续研究：

- 结构混淆强度
- GQA 对齐关系
- 安全性影响

建议继续导出：

- grouped query layout before / after permutation
- kv head layout before / after permutation
- attention map before inverse reorder
- attention map after inverse reorder

### 建议 2：补阶段 E 验收版总报告

当前建议把：

- `阶段E复杂Attention复现报告.md`
- 本文档

合并成一版“阶段 E 已修复验收报告”，正式替换最初那版“结果待排查”的表述。

---

## 15. 总结

这次阶段 E 的排错不是参数微调，而是一次真正的**结构级 bug 修复**。

修复前：

- complex attention 一加就坏；
- prefix 多层传播很快失稳；
- block 恢复链被破坏。

修复后：

- `R̂_qk / Ĥ_qk / Ẑ_block / τ_kv / τ_group` 不再破坏当前 simplified AloePri 主链；
- block0 与 prefix 多层都重新回到阶段 D 的稳定恢复水平；
- 说明当前 attention 结构版已经可以作为后续阶段继续扩展的稳定基线。

因此，这一轮工作可以正式概括为：

> **阶段 E 的核心 bug 已被定位并解决，复杂 attention 结构版已经恢复到可继续推进的状态。**
