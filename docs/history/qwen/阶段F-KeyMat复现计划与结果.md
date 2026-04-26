# 阶段 F：Algorithm 1 / KeyMat 体系复现与功能接入报告

## 1. 阶段定位

阶段 F 的目标不是继续扩层，也不是继续调复杂 attention，而是把阶段 A–E 中使用的**简化 hidden transform**，升级为技术报告 5.2.1 / 5.2.2 所要求的 **Algorithm 1 原生 KeyMat 体系**，并先在“功能正确性”上打通。

当前阶段 F 的实现采取了一个明确的工程策略：

- **矩阵生成层面**：严格实现 Algorithm 1 的 `INIT / KeyMatGen / InvKeyMatGen`
- **模型接入层面**：先采用 **projection / lift bridge 显式 wrapper 版**
- **目标**：先验证 `KeyMat` 能无缝接进现有 A–E 稳定主线，再考虑后续更接近“纯权重混淆”的 fused/server-side 形式

这意味着：

- 阶段 F 已经完成了 **“原生 KeyMat 的首次系统接入”**
- 但当前最强结论仍是 **功能性接入正确**
- 还不是最终 server-only / fully-fused 的论文终态

---

## 2. 代码交付物

### 2.1 新增模块

- `src/keymat.py`
- `src/keymat_embed_head.py`
- `src/keymat_norm.py`
- `src/keymat_ffn.py`
- `src/keymat_attention_bridge.py`
- `src/stage_f.py`

### 2.2 新增脚本

- `scripts/run_stage_f_keymat_unit.py`
- `scripts/run_stage_f_embed_head.py`
- `scripts/run_stage_f_block0.py`
- `scripts/run_stage_f_prefix_layers.py`
- `scripts/run_stage_f_full_layers.py`

### 2.3 新增测试

- `tests/test_keymat.py`
- `tests/test_keymat_embed_head.py`
- `tests/test_stage_f_block0.py`

### 2.4 新增结果文件

- `outputs/stage_f/keymat_unit.json`
- `outputs/stage_f/embed_head_eval.json`
- `outputs/stage_f/block0_eval.json`
- `outputs/stage_f/prefix_layers_2.json`
- `outputs/stage_f/prefix_layers_4.json`
- `outputs/stage_f/full_layers.json`
- `outputs/stage_f/regression_compare.json`

---

## 3. 实现摘要

## 3.1 `src/keymat.py`

这一模块实现了论文 5.2.1 的 Algorithm 1。

### 已实现对象

- `KeyMatBases`
- `KeyMatTransform`

### 已实现函数

- `init_keymat_bases(d, h, lam, seed)`
- `generate_keymat(bases, seed)`
- `generate_inv_keymat(bases, seed)`
- `build_keymat_transform(...)`
- `sample_null_columns(F_t, out_rows, seed)`
- `sample_null_rows(E, out_cols, seed)`
- `check_keymat_inverse(P, Q, tol=...)`
- `apply_keymat_transform(hidden, transform)`
- `apply_inverse_keymat_transform(hidden, transform)`

### 关键实现细节

1. `h` 被限制为正偶数，因为论文里 `E1/E2` 与 `F1/F2` 需要 `h/2`
2. 正交矩阵通过 Gaussian + QR 采样
3. `B = U + λV` 与 `B^{-1}` 使用 `float64` 构造
4. `C` 和 `D` 通过 null space basis 构造，保证：
   - `C @ F = 0`
   - `E @ D = 0`
5. 因而：
   - `P = [B C E] Z`
   - `Q = Z^T [B^{-1}; F; D]`
   - 满足 `P @ Q = I_d`

### 重要说明

论文 OCR 文本中 `C` / `D` 的“columns / rows sampled from null(...)”表述与矩阵维度有歧义；当前实现采用了与 `P @ Q = I` 一致的构造方式：

- `C` 的**行**落在 `null(F^T)` 张成空间里，从而保证 `C F = 0`
- `D` 的**列**落在 `null(E)` 张成空间里，从而保证 `E D = 0`

这是当前实现中最关键的形状解释。

---

## 3.2 `src/keymat_embed_head.py`

### 已实现功能

- embedding/head 噪声注入
- permutation 后的 embed/head obfuscation
- KeyMat 下 embedding/head 的显式 obfuscated weight 构造

### 核心函数

- `add_embed_noise(...)`
- `add_head_noise(...)`
- `obfuscate_embedding_with_keymat(...)`
- `obfuscate_head_with_keymat(...)`
- `restore_logits_from_keymat_head(...)`

### 包装器

- `KeyMatEmbeddingWrapper`
- `KeyMatHeadWrapper`

### 当前策略

1. 仍保留阶段 A 的稳健策略：
   - 只把普通词表 token 作为主要噪声/置换对象
   - special / added / tail rows 不主动打乱 tokenizer 语义
2. embedding obfuscation 采用：
   - `W_embed_obf = W_embed_stageA_noisy @ P`
3. head obfuscation 采用：
   - `W_head_obf = W_head_stageA_noisy @ Q^T`
4. head wrapper 支持两种输入：
   - 已处于 obfuscated hidden basis
   - 仍处于 base hidden，需要内部先乘 `P`

---

## 3.3 `src/keymat_norm.py`

### 已实现功能

- `estimate_kappa_for_keymat(...)`
- `KeyMatRMSNormBridge`
- `build_keymat_rmsnorm_wrapper(...)`
- `fuse_keymat_norm_into_adjacent_linear(...)`

### 当前实现方式

本阶段的重点是先打通功能，所以 norm 采用**exact bridge**：

- 输入 expanded hidden `y`
- 先用 `Q` 投影回 base hidden `x`
- 在 base space 上运行原始 RMSNorm
- 再用 `P` 抬升回 obfuscated hidden

因此：

- 当前 `κ` 已经被估计并纳入配置
- 但 forward 的主逻辑仍是 **projection / lift exact bridge**
- 这保证了阶段 F 先在功能上稳定

这也意味着：

> 当前阶段 F 的 norm 还不是论文最终“纯 obfuscated RMSNorm 融合态”，而是“KeyMat 功能桥接态”。

---

## 3.4 `src/keymat_ffn.py`

### 已实现功能

- `build_keymat_ffn_wrapper(...)`
- `obfuscate_ffn_with_keymat(...)`
- `KeyMatFFNBridge`

### 当前策略

1. hidden 输入/输出换成 `KeyMat` 的 `Q/P` bridge
2. FFN 内部仍复用阶段 C 已验证过的：
   - `Z_ffn`
   - `H_ffn`
   - gate/up 共享同一 permutation
3. 计算路径是：
   - expanded hidden `y`
   - `y @ Q -> x`
   - 在 base hidden 上运行稳定 FFN wrapper
   - 输出再 `@ P`

这与阶段 F 的“功能先行”策略完全一致。

---

## 3.5 `src/keymat_attention_bridge.py`

### 已实现功能

- `KeyMatAttentionBridge`

### 当前桥接策略

这里没有重写阶段 E 的复杂 attention 内核，而是把阶段 E 视作**稳定模块**：

- 输入 expanded hidden `y`
- `y @ Q -> x`
- 在 base hidden 上运行：
  - `simplified` attention，或
  - 阶段 E 的复杂 attention profile
- 输出再 `@ P`

当前默认 profile：

- `rqk_hqk_block_taukv_taugroup`

这保证了：

- 阶段 E 已修复的复杂 attention 结构不需要重新排错
- 阶段 F 只测试“KeyMat 是否能桥接到稳定 attention 主线”

---

## 3.6 `src/stage_f.py`

这一模块负责统一阶段 F 的接入、prefix/handoff、指标汇总与实验执行。

### 关键对象

- `LayerStageFConfig`
- `StageFRunResult`
- `StageFModel`
- `KeyMatDecoderLayerHandoff`

### 关键函数

- `calibrate_keymat_kappas(...)`
- `build_layer_stage_f_configs(...)`
- `build_stage_f_model(...)`
- `run_stage_f_single_prompt(...)`
- `aggregate_stage_f_results(...)`
- `build_default_stage_f_keymat(...)`

### 当前支持的接入形态

1. `embed/head only`
2. `block0`
3. `prefix-2`
4. `prefix-4`
5. `full layers`

### 工程上的一个重要修复

在实现过程中发现：

- baseline 上先挂的 hooks
- 会在 `prepare_stage_a_model -> deepcopy` 后被复制到 stage-F model
- 从而污染 baseline recorder

当前已经在 `build_stage_f_model(...)` 内新增 `_clear_copied_hooks(...)`，清除复制过来的 hooks。

这一步非常关键，否则 stage-F 记录会出现“baseline recorder 被 obfuscated model 写入”的隐蔽错误。

---

## 4. 数学单元实验：`outputs/stage_f/keymat_unit.json`

本实验覆盖：

- `hidden_size ∈ {128, 896}`
- `h ∈ {32, 64, 128}`
- `λ ∈ {0.1, 0.3, 1.0}`

共 12 组组合。

### 代表结果

示例一：

- `d=128, h=32, λ=0.1`
- `max_abs(PQ-I) = 2.2566e-07`
- `mean_abs(PQ-I) = 3.4832e-08`
- `spectral_norm(P) = 15.0227`
- `condition_number(P) = 17.1788`

最坏 `PQ-I` 误差：

- `d=896, h=128, λ=1.0`
- `max_abs(PQ-I) = 1.9067e-06`

最大 condition number：

- `d=896, h=32, λ=1.0`
- `condition_number(P) = 771.8164`

### 结论

- 所有测试组合都满足 `passes_tolerance = true`
- `P @ Q ≈ I` 在 `1e-6` 级别稳定成立
- `λ` 增大时 condition number 会明显升高，和论文的数值稳定性讨论一致

---

## 5. embed/head-only：`outputs/stage_f/embed_head_eval.json`

## 5.1 实验设置

测试 3 组：

1. `zero_noise`
   - `alpha_e = 0`
   - `alpha_h = 0`
2. `small_noise`
   - `alpha_e = 0.01`
   - `alpha_h = 0.01`
3. `medium_noise`
   - `alpha_e = 0.05`
   - `alpha_h = 0.05`

并采用 handoff-at-layer0 的显式 bridge 方式验证：

- embedding 先进入 KeyMat expanded space
- layer0 立即 handoff 回 base
- suffix 与 baseline 保持一致
- head 通过 obfuscated head wrapper 接入

## 5.2 结果

### `zero_noise`

- `avg_final_logits_restored_max_abs_error = 0.0003696`
- `avg_final_logits_restored_mean_abs_error = 2.70e-05`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### `small_noise`

- `avg_final_logits_restored_max_abs_error = 0.2892`
- `avg_final_logits_restored_mean_abs_error = 0.03553`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### `medium_noise`

- `avg_final_logits_restored_max_abs_error = 1.5306`
- `avg_final_logits_restored_mean_abs_error = 0.18279`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 0.8`
- `generated_text_exact_match_rate = 0.8`

## 5.3 结论

- `alpha=0` 时，KeyMat embed/head-only 已经几乎退化为阶段 A 的精确功能基线
- 小噪声下功能几乎不受影响
- 中等噪声下开始出现少量生成偏差，但仍未崩坏

这说明：

> 论文 5.2.2 的 embed/head 路径已经可以在当前工程链路中独立接入，而且不会立即破坏功能。

---

## 6. block0：`outputs/stage_f/block0_eval.json`

本实验分两种模式：

1. `embed_head_only`
2. `block0_full`

其中 `block0_full` 包含：

- KeyMat hidden bridge
- norm bridge
- attention bridge（接阶段 E 稳定 attention profile）
- FFN bridge

## 6.1 `embed_head_only`

- `avg_final_logits_restored_max_abs_error = 0.0003696`
- `avg_layer_0_input_max_abs_error = 4.41e-07`
- `avg_layer_0_block_out_max_abs_error = 1.4782e-05`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`

## 6.2 `block0_full`

- `avg_final_logits_restored_max_abs_error = 0.0004059`
- `avg_layer_0_input_norm_out_max_abs_error = 1.2290e-05`
- `avg_layer_0_q_proj_out_max_abs_error = 2.6703e-05`
- `avg_layer_0_k_proj_out_max_abs_error = 3.0518e-05`
- `avg_layer_0_v_proj_out_max_abs_error = 1.73e-06`
- `avg_layer_0_attn_out_max_abs_error = 1.8314e-06`
- `avg_layer_0_post_attn_norm_out_max_abs_error = 9.0981e-05`
- `avg_layer_0_mlp_out_max_abs_error = 5.4359e-05`
- `avg_layer_0_block_out_max_abs_error = 5.5075e-05`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`

## 6.3 结论

block0 已经不只是“能跑”，而是：

> 在 KeyMat 显式 bridge 形式下，`block0` 的 `q/k/v/attn_out/mlp_out/block_out` 都回到了接近数值误差级别。

---

## 7. prefix-2：`outputs/stage_f/prefix_layers_2.json`

### 关键结果

- `avg_final_logits_restored_max_abs_error = 0.0005118`
- `avg_layer_0_block_out_max_abs_error = 5.5075e-05`
- `avg_layer_1_block_out_max_abs_error = 1.0574e-04`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### 结论

prefix-2 仍保持近乎精确恢复，没有出现阶段 D/E 里那种“误差逐层扩散”的现象。

---

## 8. prefix-4：`outputs/stage_f/prefix_layers_4.json`

### 关键结果

- `avg_final_logits_restored_max_abs_error = 0.002122`
- `avg_layer_0_block_out_max_abs_error = 5.5075e-05`
- `avg_layer_1_block_out_max_abs_error = 1.0574e-04`
- `avg_layer_2_block_out_max_abs_error = 0.006409`
- `avg_layer_3_block_out_max_abs_error = 0.010742`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### 结论

误差开始轻微增长，但仍然非常小，且没有出现断崖式失稳。

---

## 9. full layers：`outputs/stage_f/full_layers.json`

### 关键结果

- `avg_final_logits_restored_max_abs_error = 0.003606`
- `avg_layer_23_block_out_max_abs_error = 0.003561`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### 与阶段 D / E 的系统级比较

参考 `outputs/stage_f/regression_compare.json`：

| 阶段 | `final_logits max error` | `layer_23 block_out max error` | greedy 首 token | generated ids |
|---|---:|---:|---:|---:|
| Stage D full | `20.0770` | `7.9754` | `0.8` | `0.4` |
| Stage E full | `20.0770` | `7.9754` | `0.8` | `0.4` |
| Stage F full | `0.003606` | `0.003561` | `1.0` | `1.0` |

### 结论

从“功能正确性”这个标准看，阶段 F 的显式 KeyMat bridge 结果明显强于阶段 D/E。

---

## 10. 阶段 F 的真实技术边界

这是阶段 F 最需要被明确写清的一节，因为它决定了：

- 当前结果到底代表了什么；
- 哪些模块已经是真正的 obfuscated weights；
- 哪些模块仍然依赖 runtime bridge；
- 为什么阶段 F 会得到明显优于阶段 D/E 的功能指标。

### 10.1 总结性判断

当前阶段 F 不是“整模型 fully-obfuscated weights 已完成”的状态，而是一个**混合实现**：

- **真正已经构造成 obfuscated weights 的部分**
  - `embedding`
  - `lm_head`
- **仍然主要依赖显式 `Q/P` bridge 的部分**
  - `RMSNorm`
  - `attention`
  - `FFN`
  - prefix/full-layer 的层间 handoff

因此，阶段 F 当前的最准确定义是：

> **论文原生 KeyMat 的首次系统功能接入，且采用的是“部分真实权重混淆 + 大量显式 bridge”的混合形态。**

### 10.2 哪些部分已经是“真正的混淆权重”

#### embedding

在 `src/keymat_embed_head.py` 中，embedding 权重是直接构造的：

- `obfuscate_embedding_with_keymat(...)`
- 公式实现为：`W_embed_obf = W_embed_noisy @ P`

随后在 `src/stage_f.py` 中，被真正替换成新的 embedding 模块：

- `KeyMatEmbeddingWrapper`

这意味着：

> 当前阶段 F 的 embedding 不是简单在 forward 里外挂矩阵，而是已经在使用新的 obfuscated embedding 权重。

#### model head

head 权重同样是直接构造的：

- `obfuscate_head_with_keymat(...)`
- 公式实现为：`W_head_obf = W_head_noisy @ Q^T`

随后在 `src/stage_f.py` 中被替换成：

- `KeyMatHeadWrapper`

因此：

> 当前阶段 F 的 `lm_head` 也已经是真正构造出来的 obfuscated weight，而不是保留原 head 再外挂乘法。

### 10.3 哪些部分还不是“真正的混淆权重”

#### RMSNorm

在 `src/keymat_norm.py` 中，`KeyMatRMSNormBridge` 的 forward 路径是：

1. `hidden_states @ Q -> base_hidden`
2. 在 base hidden 上运行原始 `norm_layer`
3. 输出再 `@ P`

也就是说，当前 norm 仍然是：

> **显式 bridge 版**，不是论文最终的“纯 obfuscated RMSNorm 融合态”。

另外需要明确一点：

- `κ` 已经被估计并存入模块配置；
- 但当前 forward 的主功能正确性并不是依赖“纯 κ 修正的 obfuscated norm”得到的；
- 它主要来自 `Q -> plain RMSNorm -> P` 这条 exact bridge。

所以：

> 当前阶段 F 的 norm 成功，代表 KeyMat 可以正确桥接到 norm，不代表我们已经完成了论文最终形态的 fused obfuscated RMSNorm。

#### attention

在 `src/keymat_attention_bridge.py` 中，`KeyMatAttentionBridge` 的 forward 路径是：

1. `hidden_states @ Q -> base_hidden`
2. 调用阶段 E 已经稳定的 attention 内核
3. 输出再 `@ P`

这说明当前 attention 的成功，来自：

- KeyMat 把 hidden 正确桥接回 base space；
- 阶段 E 的复杂 attention 内核本身已经被验证稳定。

因此：

> 当前阶段 F 还没有把 KeyMat 完整吸收到 `q_proj / k_proj / v_proj / o_proj` 的所有权重里。

也就是说，attention 还不是 fully-fused 的 KeyMat obfuscated attention。

#### FFN

在 `src/keymat_ffn.py` 中，`KeyMatFFNBridge` 的 forward 路径同样是：

1. `hidden_states @ Q -> base_hidden`
2. 在 base hidden 上复用阶段 C 已稳定的 FFN wrapper
3. 输出再 `@ P`

因此：

> 当前 FFN 也还不是 fully-fused 的 KeyMat 权重版本，而是 KeyMat bridge + 已稳定 FFN 内核的组合。

#### 层间 handoff

在 `src/stage_f.py` 中，`KeyMatDecoderLayerHandoff` 用于：

- 当 prefix 只适配前若干层时，
- 在 handoff 位置把 expanded hidden 拉回 base hidden，
- 交给后续未适配层继续运行。

这本质上也是一种显式桥接，而不是纯 obfuscated 模型整体自然接续。

### 10.4 为什么阶段 F 的误差会远小于阶段 D/E

这件事必须正确解读，否则很容易得出错误结论。

阶段 F 的 full-layer 结果非常好：

- `avg_final_logits_restored_max_abs_error ≈ 0.003606`
- `generated_ids_exact_match_rate = 1.0`

这并不意味着：

> “fully-fused 的 KeyMat obfuscated model 已经天然比阶段 D/E 更准确”

更准确的解释是：

> 当前阶段 F 的大量模块都在使用 **projection / lift exact bridge**：
>
> 1. `obfuscated -> base`
> 2. 在 base space 上运行已经验证正确的模块
> 3. `base -> obfuscated`

因此它本质上更接近一种**显式可逆桥接系统**，会显著压低功能误差。

这也是为什么阶段 F 现在的数值结果会明显优于阶段 D/E：

- 阶段 D/E 仍然是在“尽量让 obfuscated block 直接协变地工作”
- 阶段 F 则在大量位置采取了“先回明文 basis 再算”的 exact bridge

### 10.5 这对阶段 F 的结论意味着什么

阶段 F 已经可靠证明了：

1. Algorithm 1 的原生 KeyMat 能生成稳定互逆的 `P / Q`
2. KeyMat 能无缝接进当前 A–E 已稳定的主线
3. KeyMat 不会天然破坏：
   - embedding/head
   - norm
   - attention
   - FFN
   - residual / prefix 传播

但是，阶段 F **还没有**证明：

1. 已经完成 fully-fused 的 server-side 权重混淆
2. 已经把 `Q/P` 从 norm / attention / FFN 的 runtime bridge 中完全消掉
3. 当前误差水平等价于“纯 obfuscated 权重模型”的误差水平

因此，阶段 F 的最准确结论应写成：

> **阶段 F 已完成论文原生 KeyMat 的首次系统功能接入；其中 embedding/head 已经是真实混淆权重，而 norm/attention/FFN 仍主要通过显式 `Q/P` bridge 实现功能正确性。**

### 10.6 阶段 F 之后真正还剩什么

如果要走向“真正的混淆权重模型”，下一阶段的重点就不再是：

- 证明 KeyMat 能不能接进去

而是：

- 如何把当前 bridge 里的 `Q/P` 吸收到参数中
- 逐步减少 runtime 显式投影
- 把 norm / attention / FFN 从 bridge 版推进到 fused-weight 版

也就是说，阶段 F 之后最自然的下一阶段应当是：

> **KeyMat 融合化 / 去 bridge 化**

这是从“功能正确”走向“更像论文最终部署形态”的关键过渡。

---

## 11. 测试情况

当前测试全部通过：

- `conda run --no-capture-output -n qwen-transformers pytest -q`
- 结果：`29 passed`

新增覆盖包括：

- KeyMat 互逆与 null-space 约束
- embed/head 无噪声闭环
- stage-F block0 真实模型运行与误差上界

---

## 12. 阶段 F 的正式结论

按阶段 F 的原始目标来判定，当前结论是：

### 12.1 已完成

1. 论文 5.2.1 的 Algorithm 1 已独立实现
2. `P @ Q ≈ I` 在多组 `(d,h,λ)` 下严格成立
3. 论文 5.2.2 的 embedding/head 路径已接入
4. block0 层面，KeyMat 下的 norm + attention + FFN 已打通
5. prefix-2 / prefix-4 / full-layers 均无失控爆炸
6. full-layer 下 final logits 与生成已经回归到几乎精确一致

### 12.2 当前边界

1. 当前接入形式仍是 **显式 bridge 版**
2. 还不是 fully-fused 的 server-side 纯权重 obfuscation 形态
3. 阶段 E 的复杂 attention 仍通过稳定 wrapper/profile 复用，而不是重新在 KeyMat 下做权重融合

### 12.3 当前最准确的阶段判定

> **阶段 F 已完成“论文原生 KeyMat 体系的首次系统接入与功能验证”。**

---

## 13. 下一步建议

阶段 F 之后最合理的下一步，不再是继续证明“功能对不对”，而是进入更贴近论文后续目标的两条路线之一：

### 路线 G：安全性 / 攻击评估

既然阶段 F 的 KeyMat 已经接进来，下一步可以开始：

- token permutation 恢复难度分析
- head-level / attention-map 结构混淆强度分析
- embedding/head 噪声对恢复攻击的影响

### 路线 G'：权重融合化

如果想更靠近 server-side 最终部署形式，则下一步应做：

- 将当前 stage F 的 exact bridge 改写为更高比例的 fused-weight 版
- 逐步减少运行时 `Q/P` 显式投影
- 验证 fused 后误差与数值稳定性

---

## 14. 一句话总结

> 阶段 F 已经证明：**Algorithm 1 的原生 KeyMat 不仅能生成稳定可逆矩阵，而且能在当前 A–E 主线上实现近乎精确的系统功能恢复；当前剩余工作不再是“能不能接进去”，而是“如何把它更像论文最终部署形式地融合进去，并转入安全评估”。**
