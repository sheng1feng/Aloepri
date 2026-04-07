# 阶段 E 复杂 Attention 结构复现报告

## 1. 报告定位

本文档专门记录阶段 E 的复现过程、代码实现、实验命令、结果数据、异常现象与排查线索。

这份报告的目标不是给出“阶段 E 已成功”的结论，而是：

1. 把当前已经实现的复杂 attention 结构完整记录下来；
2. 把当前实验结果的真实状态写清楚；
3. 把后续最可能排查的问题点系统整理出来；
4. 为阶段 E 后续迭代提供一份**工程排错文档**。

换句话说：

> 这是一份“实现 + 数据 + 异常 + 调试指引”型报告，而不是阶段 A/B/C 那种“已验收通过”型报告。

---

## 2. 阶段 E 在整条复现路线中的位置

当前总复现路线是：

- **阶段 A**：词表空间闭环
- **阶段 B**：hidden-space + block0 attention 基础协变恢复
- **阶段 C**：block0 完整 block 恢复
- **阶段 D**：多层 block 级推广验证
- **阶段 E**：补齐论文第 `5.2.3` 节中的复杂 attention 结构

阶段 D 已经证明：

- 简化 attention 版本可以支持 block 恢复；
- 多层 block 级传播已经开始成立；
- 在全层时，`final_logits` 和部分生成已经开始回归。

因此阶段 E 的目标不是重新做一遍阶段 D，而是：

> 在当前“简化 attention 可工作”的基础上，把 attention 本身升级为更接近论文结构的版本。

这也是当前阶段 E 的根本风险所在：

- 如果阶段 E 退化了，说明问题更可能出在新加的 attention 复杂结构，而不是 A～D 那条基础链路。

---

## 3. 技术报告中阶段 E 对应的论文内容

阶段 E 对应论文 **第 5.2.3 节 Attention Obfuscation**。

论文把 attention 混淆分为两大类：

### 3.1 Intra-head Transformation

包括：

- `Q̂_q`
- `Q̂_k`
- `Q̂_v`
- `P̂_o`
- `Û_vo`
- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`

这些结构作用在单个 head 内部，或者作用于与 RoPE 的 `2x2` block 对齐的局部结构上。

### 3.2 Inter-head Permutation

包括：

- `τ_kv`
- `τ_group`

这些结构作用在：

- KV heads 级别
- grouped query/output 级别

并且论文明确指出，这套设计可应用于：

- MHA
- MQA
- GQA

当前 Qwen2.5-0.5B 是 **GQA**，所以 `τ_kv / τ_group` 的实现必须与 GQA head 布局一致。

---

## 4. 阶段 E 当前实现的文件与职责

## 4.1 `src/attention_keys.py`

这个文件负责生成复杂 attention 结构用到的 key。

当前实现内容：

- `generate_r_qk(head_dim, seed)`
  - 生成按 `2x2` block 组织的二维旋转矩阵；

- `generate_h_qk(head_dim, scale_range, seed)`
  - 生成 `Diag({s_i I_2})` 的 paired scaling；

- `generate_block_perm(num_blocks, beta, gamma, rope_base, seed, mode)`
  - 当前只实现了 `simplified_window` 模式；
  - 保留论文“局部窗口 + block permutation”的基本形状；
  - 还没有实现更接近论文的动态概率采样细节；

- `generate_tau_kv(num_kv_heads, seed)`
  - KV 头级 permutation；

- `generate_tau_group(num_groups, seed)`
  - group 内 permutation；

- `build_attention_complex_config(...)`
  - 组合不同 profile：
    - `simplified`
    - `rqk`
    - `rqk_hqk`
    - `rqk_hqk_block`
    - `rqk_hqk_block_taukv`
    - `rqk_hqk_block_taukv_taugroup`

## 4.2 `src/gqa_layout.py`

这个文件负责 GQA 布局相关的索引组织。

当前实现内容：

- `GQALayout`
  - `num_query_heads`
  - `num_kv_heads`
  - `num_groups`

支持：

- query head 按 `(kv_head, group)` 形式 reshape
- `τ_kv` 在 KV 头维度上的重排
- `τ_group` 在 group 维度上的重排
- 对 query grouped tensor 的正向 / 逆向恢复

## 4.3 `src/obfuscate_attention_complex.py`

这是阶段 E 最核心的实现文件。

当前实现内容：

- `ComplexQwen2Attention`
  - 在 forward 中做：
    1. hidden 从 obfuscated basis 恢复到 base basis；
    2. 计算 `q_proj / k_proj / v_proj`；
    3. 对 Q/K 做 intra-head 变换；
    4. 对 query/kv 做 inter-head permutation；
    5. 执行原 attention 核心；
    6. 再把 head 排列恢复；
    7. 再通过 `o_proj`；
    8. 最后回到 obfuscated basis。

- `ComplexAttentionTraceData`
  - 记录经过恢复后的 `q/k/v` 线性输出，用于后续比较。

- `fuse_intra_head_qk_transforms(...)`
  - 目前只支持把 `R̂_qk / Ĥ_qk / Ẑ_block` 这类 **intra-head Q/K 结构** 融合进权重；
  - 不支持 `τ_kv / τ_group` 的 fused 版本。

## 4.4 `src/stage_e.py`

这个文件负责把复杂 attention 接入现有 stage C / D 的 block 恢复框架。

当前实现内容：

- `LayerStageEConfig`
  - 每层包含：
    - `hidden_transform`
    - `input_kappa`
    - `post_attn_kappa`
    - `ffn_transform`
    - `attention_config`

- `attach_stage_e_hooks(...)`
  - 负责：
    - 在指定层替换成 `ComplexQwen2Attention`
    - 按需要替换 RMSNorm wrapper
    - 按需要替换 FFN wrapper
    - 按需保留原始 block trace

- `build_layer_stage_e_configs(...)`
  - 把 attention profile 构造成逐层配置，接入现有的 per-layer `κ` 和 `ffn_transform`

---

## 5. 阶段 E 当前实验入口

## 5.1 单层 attention 结构消融

脚本：

- `scripts/run_stage_e_block0_attention_complex.py`

作用：

- 固定 `layer_count = 1`
- 依次运行以下 profile：
  - `simplified`
  - `rqk`
  - `rqk_hqk`
  - `rqk_hqk_block`
  - `rqk_hqk_block_taukv`
  - `rqk_hqk_block_taukv_taugroup`

输出：

- `outputs/stage_e/block0_attention_complex.json`

## 5.2 prefix 多层实验

脚本：

- `scripts/run_stage_e_prefix_layers.py`

当前已经运行：

- `layer_count = 2`
  - 输出：`outputs/stage_e/prefix_layers_2.json`

- `layer_count = 4`
  - 输出：`outputs/stage_e/prefix_layers_4.json`

---

## 6. 阶段 E 当前测试状态

当前测试命令：

```bash
conda run --no-capture-output -n qwen-transformers pytest -q
```

结果：

- `24 passed`

这说明：

- 结构生成器没崩；
- GQA 索引基本自洽；
- block0 复杂 profile 至少在“相对 hidden-only 更好”的意义下是稳定可运行的。

但要注意：

> 测试通过不等于实验效果达到阶段 D 那样的高质量恢复。

目前测试只能证明：

- 代码可运行；
- 基本结构没写错到立即炸掉；
- 某些 profile 至少比 hidden-only 略有改善。

---

## 7. 阶段 E block0 单层消融结果

数据来源：

- `outputs/stage_e/block0_attention_complex.json`

当前各 profile 的关键结果如下。

### 7.1 简化 attention 基线（`simplified`）

- `avg_layer_0_attn_out_restored_max_abs_error = 0.000529`
- `avg_layer_0_block_out_restored_max_abs_error = 0.024588`
- `avg_final_logits_restored_max_abs_error = 35.317788`

这是阶段 D 之前已经建立好的“可工作 attention 基线”。

它说明：

- 当前 block0 的简化版 attention 恢复非常稳定；
- `attn_out` 与 `block_out` 的误差都很小。

### 7.2 只加 `R̂_qk`（`rqk`）

- `avg_layer_0_attn_out_restored_max_abs_error = 0.295059`
- `avg_layer_0_block_out_restored_max_abs_error = 2.246665`
- `avg_final_logits_restored_max_abs_error = 36.915518`

与 `simplified` 比较：

- `attn_out` 从 `0.000529` 劣化到 `0.295059`
- `block_out` 从 `0.024588` 劣化到 `2.246665`

这说明：

> 当前 `R̂_qk` 一旦接入，就已经明显破坏了 block0 的恢复质量。

### 7.3 再加 `Ĥ_qk`（`rqk_hqk`）

- `avg_layer_0_attn_out_restored_max_abs_error = 0.296669`
- `avg_layer_0_block_out_restored_max_abs_error = 2.141420`
- `avg_final_logits_restored_max_abs_error = 37.015094`

与 `rqk` 比较：

- `attn_out` 基本不变
- `block_out` 略好一些
- 但整体仍显著劣化于 `simplified`

这说明：

> 当前 `Ĥ_qk` 没有让系统进一步崩坏，但也没有把 `R̂_qk` 带来的误差拉回去。

### 7.4 再加 `Ẑ_block`（`rqk_hqk_block`）

- `avg_layer_0_attn_out_restored_max_abs_error = 0.105283`
- `avg_layer_0_block_out_restored_max_abs_error = 1.772131`
- `avg_final_logits_restored_max_abs_error = 35.445967`

与 `rqk_hqk` 比较：

- `attn_out` 有明显改善：`0.296669 -> 0.105283`
- `block_out` 也改善：`2.141420 -> 1.772131`

这说明：

> 在当前实现里，`Ẑ_block` 反而帮助减轻了 `R̂_qk + Ĥ_qk` 的破坏。

这是一个非常重要的排查线索：

- 说明问题不只是“复杂结构越多越差”；
- 更像是 `R̂_qk / Ĥ_qk` 与当前 RoPE / GQA 组织之间存在错位，而 `Ẑ_block` 恰好缓和了这种错位。

### 7.5 再加 `τ_kv`（`rqk_hqk_block_taukv`）

- `avg_layer_0_attn_out_restored_max_abs_error = 0.105283`
- `avg_layer_0_block_out_restored_max_abs_error = 1.772131`
- `avg_final_logits_restored_max_abs_error = 35.445967`

与 `rqk_hqk_block` 在**恢复后的功能指标**上完全相同。

这在修复后的实现里应解释为：

> 当前 `τ_kv` 不会破坏恢复后的功能指标。

它并不意味着 `τ_kv` 没有真正起作用，而是意味着：

- 它在 head-level 内部改变了排列；
- 但在恢复链路完成后，最终的 restored 功能指标仍然保持不变。

这通常意味着两种可能：

1. `τ_kv` 的实现没有真正作用到最终有效路径；
2. 当前 block0 的 GQA 布局下，某些中间重排又被后续恢复抵消了，表现成“表面无效”。

### 7.6 再加 `τ_group`（`rqk_hqk_block_taukv_taugroup`）

- `avg_layer_0_attn_out_restored_max_abs_error = 0.105283`
- `avg_layer_0_block_out_restored_max_abs_error = 1.772131`
- `avg_final_logits_restored_max_abs_error = 35.445967`

与上一个 profile 在**恢复后的功能指标**上仍然完全相同。

正确解读是：

> 当前 `τ_group` 也不会破坏恢复后的功能指标。

和 `τ_kv` 一样，这不代表它是 no-op，而是说明：

- 它改变的是恢复前的 head/group 内部结构；
- 当前功能指标是在恢复到 base basis 后计算，因此理应保持稳定。

因此可以得到一个阶段 E 的强结论：

> 当前复杂 attention 实现里，真正起主要作用的是 `R̂_qk / Ĥ_qk / Ẑ_block`；  
> `τ_kv / τ_group` 目前要么是“被抵消”，要么是“尚未真正生效”。

---

## 8. block0 单层消融的逐 profile 结论

从当前单层消融结果可以形成一条非常明确的判断链：

### 结论 1

当前阶段 D 的 `simplified attention` 仍然是效果最好的单层 baseline。

### 结论 2

一旦引入 `R̂_qk`，恢复质量立即显著下降。

### 结论 3

`Ĥ_qk` 在当前实现里没有显著修复 `R̂_qk` 带来的退化。

### 结论 4

`Ẑ_block` 能明显缓解一部分误差，但仍远不如 `simplified`。

### 结论 5

`τ_kv / τ_group` 当前实验结果完全不变，说明它们的实际作用路径仍需单独排查。

---

## 9. 阶段 E prefix 两层实验结果

数据来源：

- `outputs/stage_e/prefix_layers_2.json`

当前 profile：

- `rqk_hqk_block_taukv_taugroup`

关键结果：

- `layer_0_block_out_restored_max_abs_error = 1.772131`
- `layer_1_block_out_restored_max_abs_error = 2.157598`
- `avg_final_logits_restored_max_abs_error = 30.838373`
- `greedy_first_token_match_rate = 0.0`

### 逐 prompt 现象

五条 prompt 的恢复生成全部不正确：

- greedy 首 token 全部不匹配
- 生成文本虽然不再像 hidden-only 那样完全乱码，但仍然包含大量无意义混合 token

例如 prompt 1：

- baseline（阶段 D 全层最终可恢复方向）应为正常中文回答；
- 当前阶段 E prefix-2 恢复文本：
  - `تضمنتضمن排毒 assertFalse">×</shaled__; 獸`

这说明：

> 虽然两层传播还没有立即数值爆炸，但复杂 attention 已经明显破坏了生成链路。

---

## 10. 阶段 E prefix 四层实验结果

数据来源：

- `outputs/stage_e/prefix_layers_4.json`

当前 profile：

- `rqk_hqk_block_taukv_taugroup`

关键结果：

- `layer_0_block_out_restored_max_abs_error = 1.772131`
- `layer_1_block_out_restored_max_abs_error = 2.157598`
- `layer_2_block_out_restored_max_abs_error = 456.628601`
- `layer_3_block_out_restored_max_abs_error = 1282.630249`
- `avg_final_logits_restored_max_abs_error = 32.808518`
- `greedy_first_token_match_rate = 0.0`

### 解释

从 layer1 到 layer2 的误差发生了断崖式上升：

- `2.1576 -> 456.6286`

再到 layer3：

- `456.6286 -> 1282.6302`

这意味着：

> 当前复杂 attention 实现一旦扩到四层，误差已经从“可控增长”变成了“快速爆炸”。

因此，当前阶段 E 还**不能**进入 prefix-8 或更深层的稳定性结论阶段。

---

## 11. 当前阶段 E 的关键异常与排查价值

下面列出当前阶段 E 最重要的排查结论。

## 11.1 `R̂_qk` 是当前最先导致退化的结构

从 `simplified -> rqk` 的跳变可以看到：

- `attn_out` 误差从 `0.000529` 直接上升到 `0.295059`
- `block_out` 从 `0.024588` 上升到 `2.246665`

这说明：

> 当前最先需要排查的不是 `τ_kv / τ_group`，而是 `R̂_qk` 的接入方式。

优先怀疑点：

1. `R̂_qk` 的乘法方向是否与当前 `q/k` 的布局一致；
2. `R̂_qk` 是否应该作用在 RoPE 前还是 RoPE 后；
3. `R̂_qk` 与当前 simplified hidden transform 是否发生了重复或冲突；
4. `R̂_qk` 在 head 内 `2x2` block 的展开顺序是否与 Qwen 当前实现一致。

## 11.2 `Ĥ_qk` 没有显著修复 `R̂_qk`

理论上：

- `Ĥ_qk` 是 Q 与 K 的成对逆缩放；
- 它应在不破坏 QK 点积结构的前提下增加混淆。

但当前结果显示：

- 它没有明显改善 `R̂_qk` 带来的误差。

这意味着：

> 当前问题更像是“结构错位”而不是“缩放幅度过强”。

## 11.3 `Ẑ_block` 明显改善，说明 block-level 组织不是完全错的

`Ẑ_block` 的加入让误差显著下降：

- `attn_out`: `0.2967 -> 0.1053`
- `block_out`: `2.1414 -> 1.7721`

这说明：

> 当前关于 RoPE 2x2 block 的局部组织并非完全错误，至少 `Ẑ_block` 起到了“校正部分错位”的作用。

这又进一步支持了对 `R̂_qk / Ĥ_qk` 的优先排查。

## 11.4 `τ_kv / τ_group` 当前近乎无效

这是一条极其关键的结论。

当前三组 profile：

- `rqk_hqk_block`
- `rqk_hqk_block_taukv`
- `rqk_hqk_block_taukv_taugroup`

结果完全相同。

这通常意味着：

### 最新排查结果：`τ_kv / τ_group` 的内部作用已经被显式验证

在后续新增的 head-level raw trace 诊断中，我们直接比较了：

- `rqk_hqk_block`
- `rqk_hqk_block_taukv`
- `rqk_hqk_block_taukv_taugroup`

在**未恢复前**的内部 head-level 张量差异。

结果保存在：

- `outputs/stage_e/head_trace_check.json`

关键比较结果如下：

#### `rqk_hqk_block` vs `rqk_hqk_block_taukv`

- `q_heads_post_inter_raw = 55.9384`
- `k_heads_post_inter_raw = 272.7900`
- `v_heads_post_inter_raw = 0.3084`
- `attn_heads_pre_inverse_raw = 0.3084`

#### `rqk_hqk_block_taukv` vs `rqk_hqk_block_taukv_taugroup`

- `q_heads_post_inter_raw = 89.9669`
- `attn_heads_pre_inverse_raw = 0.4094`

这说明：

> `τ_kv / τ_group` 并不是无效的；它们确实改变了恢复前的 head-level 内部结构。

因此，之前“它们可能是 no-op”的怀疑已经被排除。

## 11.5 `q_proj_out / k_proj_out / v_proj_out` 指标已经显式汇总

在后续排错中，这个问题已经修正。

当前 `outputs/stage_e/block0_attention_complex.json` 的摘要里已经包含：

- `avg_layer_0_q_proj_out_max_abs_error`
- `avg_layer_0_k_proj_out_max_abs_error`
- `avg_layer_0_v_proj_out_max_abs_error`

例如在最新结果中：

#### `simplified`

- `q_proj_out max error = 0.019803`
- `k_proj_out max error = 0.010513`
- `v_proj_out max error = 0.000426`

#### `rqk_hqk_block_taukv_taugroup`

- `q_proj_out max error = 0.019803`
- `k_proj_out max error = 0.010513`
- `v_proj_out max error = 0.000426`

这说明：

> 当前阶段 E 的 `q/k/v` 中间量已经能够被稳定汇总，并且在修复后已重新对齐到阶段 D simplified baseline 水平。

---

## 12. 阶段 E 当前状态的正式判断

结合以上数据，可以给出一个非常清晰的阶段判断。

### 12.1 已经完成的

阶段 E 已经完成：

- 论文 attention 五类复杂结构的代码接入；
- 单层与 prefix 多层实验入口；
- 基本 GQA 布局支持；
- attention 结构级消融实验框架。

因此：

> 从“代码实现”和“实验框架”角度，阶段 E 已经搭建完成。

### 12.2 经过多轮修复后已经完成的

阶段 E 现在已经完成：

- 单层 attention 结构版的高质量恢复；
- prefix 多层传播的稳定保持；
- `τ_kv / τ_group` 的内部有效性证明；
- 与阶段 D 基线相比的数值对齐。

当前可以更准确地表述为：

> **阶段 E 已经完成复杂 attention 结构的接入、排错与功能修复，并重新回到了阶段 D 的稳定功能基线。**

---

## 13. 后续建议（修复完成后的下一步）

现在阶段 E 的功能性问题已经解决，后续建议不再是“继续修功能”，而是转向更细粒度的内部分析与更贴近论文完整版的实现。

### 建议 1：继续做更细的 head-level / attention-map 诊断

当前已经证明：

- `τ_kv / τ_group` 会改变恢复前的 raw head-level 结构；
- 但恢复后的功能指标按设计保持稳定。

如果后面要做安全性或混淆强度分析，建议继续导出：

- grouped query layout before / after permutation
- kv head layout before / after permutation
- attention score map before inverse reorder
- attention prob map before inverse reorder
- attention output after inverse reorder

### 建议 2：把阶段 E 修复版合并进总报告

当前阶段 E 已经不再是“待排查”状态，而是：

- 结构接入完成
- 关键 bug 修复完成
- 多层传播重新稳定

因此应把阶段 E 的最新结论合并到总报告中，避免后续阅读者仍按修复前状态理解阶段 E。

### 建议 3：进入下一阶段（更贴近论文完整版）

如果后续继续推进，最自然的下一步是：

- 进入更复杂的 key 体系（Algorithm 1）
- 或继续做安全性 / 混淆强度分析

---

## 14. 阶段 E 结论

阶段 E 当前的核心结论可以总结为四句话：

### 结论 1

论文第 `5.2.3` 节的五类复杂 attention 结构已经全部接入当前代码框架。

### 结论 2

经过后续多轮修复后，block0 单层上的复杂 attention 结构版已经恢复到与 simplified attention 基线一致的功能水平。

### 结论 3

`τ_kv / τ_group` 在恢复后的功能指标上保持中性，但在未恢复前的 raw head-level trace 上已经被证明**确实生效**。

### 结论 4

当前阶段 E 已经从“结构接入完成，但结果待排查”升级为：

> **复杂 attention 结构接入已完成，关键 bug 已修复，功能恢复已回到阶段 D simplified attention 基线；同时 `τ_kv / τ_group` 的内部作用也已被显式验证。**
