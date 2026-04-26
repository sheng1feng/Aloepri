# 阶段 G：KeyMat 融合化 / 去 bridge 化实施与验收报告

## 1. 阶段 G 的目标

阶段 F 已经证明：

- Algorithm 1 / KeyMat 能稳定生成 `P / Q`
- embedding / head 已经是真实混淆权重
- block0、prefix、多层、full-layer 都能在 bridge 版下稳定工作

但阶段 F 的内部层仍主要依赖：

\[
y \rightarrow Q \rightarrow \text{plain / stable module} \rightarrow P \rightarrow y'
\]

这样的 runtime bridge。

因此阶段 G 的目标是：

> **把阶段 F 中依赖 runtime `Q/P` bridge 的内部层，逐步改造成参数融合后的 obfuscated 模块。**

本阶段按四步推进：

1. G1：RMSNorm fused
2. G2：FFN fused
3. G3：Attention fused
4. G4：系统级回归测试

---

## 2. 本阶段新增代码

### 2.1 新增模块

- `src/stage_g_norm.py`
- `src/stage_g_ffn.py`
- `src/stage_g_attention.py`
- `src/stage_g.py`

### 2.2 新增脚本

- `scripts/run_stage_g_regression.py`
- `scripts/run_stage_g_norm_block0.py`
- `scripts/run_stage_g_ffn_block0.py`
- `scripts/run_stage_g_attention_block0.py`

### 2.3 新增测试

- `tests/test_stage_g.py`

### 2.4 新增结果文件

- `outputs/stage_g/norm_block0.json`
- `outputs/stage_g/ffn_block0.json`
- `outputs/stage_g/attention_block0.json`
- `outputs/stage_g/ffn_prefix_2.json`
- `outputs/stage_g/prefix_layers_2.json`
- `outputs/stage_g/prefix_layers_4.json`
- `outputs/stage_g/full_layers.json`
- `outputs/stage_g/regression_compare.json`

---

## 3. G1：RMSNorm fused

## 3.1 初版问题

第一版 G1 按论文直觉尝试了“纯 scalar `κ` 补偿”的 fused RMSNorm：

- 在 obfuscated hidden 上直接做无权重 RMSNorm
- 再乘 `κ`
- 再把原 `Wnorm` 融到后续 attention / FFN

这一版的 block0 结果明显失稳：

- `avg_layer_0_input_norm_out_max_abs_error ≈ 1.0764`
- `avg_layer_0_q_proj_out_max_abs_error ≈ 4.9613`
- `avg_final_logits_restored_max_abs_error ≈ 8.0871`
- `generated_ids_exact_match_rate = 0.0`

这说明：

> 对当前 Algorithm 1 生成的 KeyMat 来说，单个标量 `κ` 不足以补偿 norm 扭曲。

## 3.2 修正方案

随后将 fused norm 改成了**预计算二次型的 exact fused norm**：

设：

\[
Q \in \mathbb{R}^{(d+2h)\times d}
\]

则 base RMS 可由 obfuscated hidden 直接计算：

\[
\|x\|^2 = \tilde{x} Q Q^T \tilde{x}^T
\]

因此 fused norm 不再做完整：

\[
Q \rightarrow \text{RMSNorm} \rightarrow P
\]

而是只在 obfuscated space 中使用预计算 metric：

\[
M = QQ^T
\]

来直接计算 base-space RMS，再输出：

\[
\tilde{x}_{norm} = \frac{\tilde{x}}{\mathrm{RMS}_{base}(\tilde{x})}
\]

同时保留：

- `Wnorm` 融入后续线性层
- 记录时通过 `QWnorm` 把 fused norm 输出恢复回 baseline norm 输出空间

对应实现：

- `src/stage_g_norm.py`

## 3.3 G1 block0 结果

结果文件：

- `outputs/stage_g/norm_block0.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 0.0004432`
- `avg_layer_0_input_norm_out_max_abs_error = 1.2290e-05`
- `avg_layer_0_block_out_max_abs_error = 5.5075e-05`
- `generated_ids_exact_match_rate = 1.0`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0007758`
- `avg_layer_0_input_norm_out_max_abs_error = 2.8229e-05`
- `avg_layer_0_q_proj_out_max_abs_error = 2.26498e-04`
- `avg_layer_0_attn_out_max_abs_error = 7.2241e-06`
- `avg_layer_0_post_attn_norm_out_max_abs_error = 4.5433e-04`
- `avg_layer_0_mlp_out_max_abs_error = 1.9493e-04`
- `avg_layer_0_block_out_max_abs_error = 1.9526e-04`
- `generated_ids_exact_match_rate = 1.0`

## 3.4 G1 结论

修正后的 G1 已满足目标：

- 不再依赖显式 `Q -> norm -> P`
- block0 的 norm 输出与 bridge 基线保持同量级
- 整体功能保持稳定

这意味着：

> **阶段 G 已完成第一个 fused 内部模块：RMSNorm。**

---

## 4. G2：FFN fused

## 4.1 设计

G2 保留了阶段 C 已验证稳定的中间维结构：

- `Z_ffn`
- `H_ffn`

只把 hidden 输入/输出侧的 KeyMat bridge 吃进权重：

- `gate_proj`
- `up_proj`
- `down_proj`

也就是说，FFN 不再显式做：

\[
Q \rightarrow \text{FFN} \rightarrow P
\]

而是直接使用 fused 权重计算。

对应实现：

- `src/stage_g_ffn.py`

## 4.2 G2 block0 结果

结果文件：

- `outputs/stage_g/ffn_block0.json`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0007796`
- `avg_layer_0_input_norm_out_max_abs_error = 2.8229e-05`
- `avg_layer_0_q_proj_out_max_abs_error = 2.26498e-04`
- `avg_layer_0_attn_out_max_abs_error = 7.2241e-06`
- `avg_layer_0_post_attn_norm_out_max_abs_error = 4.5433e-04`
- `avg_layer_0_mlp_out_max_abs_error = 1.9631e-04`
- `avg_layer_0_block_out_max_abs_error = 1.9617e-04`
- `generated_ids_exact_match_rate = 1.0`

## 4.3 G2 prefix-2 结果

结果文件：

- `outputs/stage_g/ffn_prefix_2.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 0.0005118`
- `avg_layer_1_block_out_max_abs_error = 1.0574e-04`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0009170`
- `avg_layer_0_block_out_max_abs_error = 1.9617e-04`
- `avg_layer_1_block_out_max_abs_error = 2.7847e-04`
- `generated_ids_exact_match_rate = 1.0`

## 4.4 G2 结论

G2 fused FFN 的核心结论是：

- block0 保持稳定
- prefix-2 没有爆炸
- 误差有轻微上升，但仍然远低于阶段 D/E 的多层量级

因此：

> **阶段 G 已完成第二个 fused 内部模块：FFN。**

---

## 5. G3：Attention fused

## 5.1 设计

G3 没有重写阶段 E 的复杂 attention 内核，而是只把 KeyMat 的外层 bridge 吃进：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

同时保持阶段 E 的复杂 attention 结构继续工作：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`
- `τ_kv`
- `τ_group`

也就是说，阶段 G 的 attention fused 目标是：

> **只去掉外层 `Q/P` bridge，不重做阶段 E 的 attention 内核。**

对应实现：

- `src/stage_g_attention.py`

## 5.2 G3 block0 结果

结果文件：

- `outputs/stage_g/attention_block0.json`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0007110`
- `avg_layer_0_input_norm_out_max_abs_error = 2.8229e-05`
- `avg_layer_0_q_proj_out_max_abs_error = 2.4176e-04`
- `avg_layer_0_attn_out_max_abs_error = 6.3062e-06`
- `avg_layer_0_post_attn_norm_out_max_abs_error = 4.5013e-04`
- `avg_layer_0_mlp_out_max_abs_error = 1.7223e-04`
- `avg_layer_0_block_out_max_abs_error = 1.7185e-04`
- `generated_ids_exact_match_rate = 1.0`

## 5.3 G3 prefix-2 结果

结果文件：

- `outputs/stage_g/prefix_layers_2.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 0.0005118`
- `avg_layer_1_block_out_max_abs_error = 1.0574e-04`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0008004`
- `avg_layer_0_block_out_max_abs_error = 1.7185e-04`
- `avg_layer_1_block_out_max_abs_error = 3.1602e-04`
- `generated_ids_exact_match_rate = 1.0`

## 5.4 G3 结论

attention fused 后：

- block0 仍然稳定
- prefix-2 仍可控
- 没有回到阶段 E 早期那种“单层看着能算、多层直接炸”的状态

因此：

> **阶段 G 已完成第三个 fused 内部模块：Attention。**

---

## 6. G4：系统级回归

## 6.1 prefix-4

结果文件：

- `outputs/stage_g/prefix_layers_4.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 0.002122`
- `avg_layer_2_block_out_max_abs_error = 0.006409`
- `avg_layer_3_block_out_max_abs_error = 0.010742`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.001789`
- `avg_layer_2_block_out_max_abs_error = 0.009033`
- `avg_layer_3_block_out_max_abs_error = 0.012451`
- `generated_ids_exact_match_rate = 1.0`

## 6.2 full-layer

结果文件：

- `outputs/stage_g/full_layers.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 0.003606`
- `avg_layer_23_block_out_max_abs_error = 0.003561`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.003537`
- `avg_layer_23_block_out_max_abs_error = 0.003760`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

## 6.3 G4 结论

系统级结果表明：

- 去 bridge 化后并没有破坏阶段 F 的系统稳定性
- prefix-4 / full-layer 都保持了和 bridge 基线同级别的恢复能力
- full-layer 下 final logits 甚至略优于当前 bridge baseline

因此：

> **阶段 G 已完成系统级回归验证。**

---

## 7. 测试情况

新增测试：

- `tests/test_stage_g.py`

已运行：

- `conda run --no-capture-output -n qwen-transformers pytest -q tests/test_stage_g.py`
- `conda run --no-capture-output -n qwen-transformers pytest -q`

结果：

- `2 passed`
- `31 passed`

---

## 8. 阶段 G 的正式结论

阶段 G 当前可以正式判定为：

1. G1：已完成 fused RMSNorm
2. G2：已完成 fused FFN
3. G3：已完成 fused Attention
4. G4：已完成 prefix / full-layer 系统级回归

并且当前最关键的结果是：

> **norm / attention / FFN 已经不再依赖阶段 F 的 runtime `Q/P` bridge 才能工作。**

这意味着项目状态已经从：

> Algorithm 1 / KeyMat 能接进系统，但内部层仍主要依赖显式 bridge

升级为：

> **KeyMat 已经实质性地吸收到核心模块参数与模块结构中，模型显著更接近论文意义上的 fully-obfuscated internal modules。**

---

## 9. 仍然保留的边界

虽然阶段 G 已经完成，但仍需准确说明当前边界：

1. 阶段 G 的 fused norm 使用了预计算二次型 `QQ^T`，它不是论文 5.2.5 最简洁的 `κI` 近似形态，而是一个更精确的 obfuscated norm 实现；
2. 阶段 E 的复杂 attention 内核仍被保留为运行时结构，没有把所有 intra/inter-head 结构都进一步压缩成单一静态权重张量；
3. 当前还没有进入论文第 6 节口径的攻击评估阶段。

---

## 10. 一句话总结

> **阶段 G 已经把阶段 F 的 KeyMat 从“runtime bridge 机制”推进成了“参数级融合机制”，并在 block0、prefix-2、prefix-4、full-layer 上验证了系统仍保持稳定恢复。**
