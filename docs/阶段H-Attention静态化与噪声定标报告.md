# 阶段 H：Attention 静态化收敛与噪声强度定标报告

## 1. 阶段 H 的范围

阶段 H 按约束只做两件事：

1. **H2：噪声强度定标**
2. **H1：Attention 静态化收敛**

本阶段明确**不做**：

- VMA / IA / ISA / IMA
- 隐私参数扫描
- 攻击恢复率统计
- accuracy-privacy tradeoff 曲线

因此阶段 H 的目标不是新增方法主线，而是做两类收口：

- **参数工作点收口**
- **部署形态收口**

---

## 2. 本阶段新增代码

### 2.1 模块

- `src/stage_h_noise.py`
- `src/stage_h_attention_static.py`
- `src/stage_h.py`

### 2.2 脚本

- `scripts/run_stage_h_noise_calibration.py`
- `scripts/run_stage_h_attention_static.py`
- `scripts/run_stage_h_joint_regression.py`

### 2.3 测试

- `tests/test_stage_h.py`

### 2.4 输出

- `outputs/stage_h/noise_calibration.json`
- `outputs/stage_h/attention_static_block0.json`
- `outputs/stage_h/attention_static_prefix_2.json`
- `outputs/stage_h/attention_static_prefix_4.json`
- `outputs/stage_h/full_layers.json`
- `outputs/stage_h/regression_compare.json`

---

## 3. H2：噪声强度定标

## 3.1 为什么先做 H2

当前 A–G 主线已经稳定，所以阶段 H 先要回答：

> 在当前实现里，`αe / αh / λ` 应该怎么设，既贴近论文，又不破坏功能？

因此 H2 只从**功能稳定性**维度评估：

- `final_logits_restored_max_abs_error`
- `layer_23_block_out_max_abs_error`
- `greedy_first_token_match_rate`
- `generated_ids_exact_match_rate`
- `generated_text_exact_match_rate`

## 3.2 测试点

### 论文默认点

- `αe = 1.0`
- `αh = 0.2`
- `λ = 0.3`
- `h = 128`
- `β = 8`
- `γ = 1e3`

### 粗扫与细定点

本阶段实际评估了：

- `stable_reference`
- `paper_default`
- `fine_a`
- `fine_b`
- `fine_c`
- `mild_a`
- `mild_b`
- `mild_c`
- `paper_lambda_low`
- `paper_lambda_high`

其中关键候选为：

- `fine_a = (αe=0.1, αh=0.05, λ=0.3, h=128, β=8, γ=1e3)`

## 3.3 H2 结果

结果文件：

- `outputs/stage_h/noise_calibration.json`

### `stable_reference`

- `avg_final_logits_restored_max_abs_error = 0.004618`
- `avg_layer_23_block_out_max_abs_error = 0.003872`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### 论文默认点 `paper_default`

- `avg_final_logits_restored_max_abs_error = 19.7789`
- `avg_layer_23_block_out_max_abs_error = 27.7285`
- `greedy_first_token_match_rate = 0.4`
- `generated_ids_exact_match_rate = 0.0`
- `generated_text_exact_match_rate = 0.0`

### 推荐非零工作点 `fine_a`

- `avg_final_logits_restored_max_abs_error = 2.6765`
- `avg_layer_23_block_out_max_abs_error = 1.7660`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 0.8`
- `generated_text_exact_match_rate = 0.8`

### 排名结果

当前自动排序结果为：

1. `stable_reference`
2. `fine_a`
3. `mild_a`
4. `fine_b`
5. `fine_c`
6. `mild_b`
7. `mild_c`
8. `paper_lambda_high`
9. `paper_lambda_low`
10. `paper_default`

## 3.4 H2 结论

H2 给出了两个明确结论：

### 结论 1：论文默认点当前不够稳

在当前实现和当前模型规模下，论文默认参数点：

- 功能误差明显偏大
- 生成恢复率为 `0`

因此它目前不能作为后续阶段的默认工作点。

### 结论 2：存在一个可复用的非零稳定工作点

当前建议的非零稳定工作点为：

- `αe = 0.1`
- `αh = 0.05`
- `λ = 0.3`
- `h = 128`
- `β = 8`
- `γ = 1e3`

这组参数保留了：

- 非零噪声
- 论文默认的 `λ / h / β / γ` 主结构设定

同时还能保持：

- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 0.8`
- `generated_text_exact_match_rate = 0.8`

因此，阶段 H 后续的 attention 静态化回归都固定在这组参数上进行。

---

## 4. H1：Attention 静态化收敛

## 4.1 当前差异

阶段 G 已经把 attention 做成 fused，但仍保留了阶段 E 的复杂 attention 内核运行时结构：

- runtime 内仍显式执行：
  - intra-head `q_matrix / k_matrix`
  - inter-head `τ_kv / τ_group`
  - 输出前的 inverse head permutation

阶段 H 的目标是：

> 尽量把这些固定结构前移到离线构造里，让 runtime 只保留标准 attention 必需部分。

## 4.2 H1 静态化策略

在 `src/stage_h_attention_static.py` 中，当前静态化策略是：

### A 类：保留在线

- q/k/v 投影后的张量乘法
- RoPE
- softmax
- attention 聚合
- o_proj

### B 类：前移到离线权重表达

- intra-head `q_matrix`
- intra-head `k_matrix`
- `τ_kv`
- `τ_group`
- `o_proj` 输入侧对应的 inverse head permutation

换句话说，当前 runtime 不再显式做：

- `_apply_intra_head(...)`
- `_apply_inter_head(...)`
- `_invert_inter_head(...)`

而是把这些结构直接吸收到：

- `q_weight`
- `k_weight`
- `v_weight`
- `o_weight`

中。

## 4.3 H1 排错过程

H1 不是一次成功的，经历了两轮关键排查。

### 第一轮失败

初版静态 attention 明显失稳：

- `avg_layer_0_attn_out_max_abs_error ≈ 0.143`
- `avg_layer_0_block_out_max_abs_error ≈ 4.47`
- prefix-4 / full-layer 直接退化

### 根因 1：比较配置不一致

最初 `staticized_candidate` 使用了 `seed + 50`，导致：

- permutation / noise / keymat / attention key 不是同一组配置

这使得 bridge baseline 和 static candidate 根本不在同一个混淆实例上比较。

修复后，bridge / static 使用完全相同的 seed。

### 第二轮失败

统一 seed 后，`q/k/v` 已与 bridge 基线基本对齐，但：

- `attn_out`
- `block_out`

仍明显劣化。

### 根因 2：`o_weight` 吸收 query-head permutation 的方向写反

进一步做最小诊断后发现：

- `q/k/v` 的静态化已经基本正确
- `o_weight` 输入列重排方向错误

修复方式是把：

- `q_feature_inv_order`

改为：

- `q_feature_order`

来吸收到 `o_weight` 的输入列布局中。

这一修复完成后，staticized attention 才真正回到了 bridge baseline 同量级。

---

## 5. H1 block0 结果

结果文件：

- `outputs/stage_h/attention_static_block0.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 3.4439`
- `avg_layer_0_attn_out_max_abs_error = 0.0177312`
- `avg_layer_0_block_out_max_abs_error = 0.1363455`
- `generated_ids_exact_match_rate = 0.6`

### staticized candidate

- `avg_final_logits_restored_max_abs_error = 3.4399`
- `avg_layer_0_attn_out_max_abs_error = 0.0177317`
- `avg_layer_0_block_out_max_abs_error = 0.1362447`
- `generated_ids_exact_match_rate = 0.6`

## 5.1 结论

block0 已经说明：

> Attention 静态化后，恢复能力没有明显劣化于阶段 G fused baseline。

---

## 6. H1 prefix-2 / prefix-4 / full-layer 结果

## 6.1 prefix-2

结果文件：

- `outputs/stage_h/attention_static_prefix_2.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 3.4437`
- `avg_layer_1_block_out_max_abs_error = 0.1312623`
- `generated_ids_exact_match_rate = 0.6`

### staticized candidate

- `avg_final_logits_restored_max_abs_error = 3.4397`
- `avg_layer_1_block_out_max_abs_error = 0.1311570`
- `generated_ids_exact_match_rate = 0.6`

## 6.2 prefix-4

结果文件：

- `outputs/stage_h/attention_static_prefix_4.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 3.4460`
- `avg_layer_2_block_out_max_abs_error = 0.2010672`
- `avg_layer_3_block_out_max_abs_error = 0.2525876`

### staticized candidate

- `avg_final_logits_restored_max_abs_error = 3.4417`
- `avg_layer_2_block_out_max_abs_error = 0.2011677`
- `avg_layer_3_block_out_max_abs_error = 0.2522361`

## 6.3 full-layer

结果文件：

- `outputs/stage_h/full_layers.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 3.4455`
- `avg_layer_23_block_out_max_abs_error = 1.6600623`
- `generated_ids_exact_match_rate = 0.6`
- `generated_text_exact_match_rate = 0.6`

### staticized candidate

- `avg_final_logits_restored_max_abs_error = 3.4421`
- `avg_layer_23_block_out_max_abs_error = 1.6673660`
- `generated_ids_exact_match_rate = 0.6`
- `generated_text_exact_match_rate = 0.6`

## 6.4 结论

H1 的系统级结论是：

- prefix-2 没有劣化
- prefix-4 没有劣化
- full-layer 没有劣化
- staticized attention 与阶段 G bridge baseline 处于同量级

因此：

> **attention 的 runtime 复杂结构已经成功进一步前移到离线权重表达中。**

---

## 7. 当前 H1 的“在线保留 / 离线固化”边界

### 已离线固化

- q/k 的 intra-head 线性变换
- `τ_kv`
- `τ_group`
- `o_proj` 输入侧对应的 inverse permutation

### 仍必须在线保留

- RoPE
- softmax
- attention score / probability 计算
- attention 聚合

这和论文第 5.2.3 节的精神是一致的：

> 尽量把 obfuscation 结构固化到权重表达中，但 attention 作为运算本身仍然在线执行。

---

## 8. 混淆模型交付与 client/server 划分

阶段 H 已经导出一版更贴近当前最新实现的混淆模型工件：

- `artifacts/stage_h_full_obfuscated/metadata.json`
- `artifacts/stage_h_full_obfuscated/model_state.pt`
- `artifacts/stage_h_full_obfuscated/server_model_state.pt`
- `artifacts/stage_h_full_obfuscated/client_secret.pt`

其中：

- `server_model_state.pt ≈ 4.1G`
- `client_secret.pt ≈ 2.4M`

此外，当前还新增了一套更接近 `save_pretrained` 的导出目录：

- `artifacts/stage_h_pretrained/server/config.json`
- `artifacts/stage_h_pretrained/server/generation_config.json`
- `artifacts/stage_h_pretrained/server/tokenizer.json`
- `artifacts/stage_h_pretrained/server/tokenizer_config.json`
- `artifacts/stage_h_pretrained/server/chat_template.jinja`
- `artifacts/stage_h_pretrained/server/model.safetensors`
- `artifacts/stage_h_pretrained/server/obfuscation_config.json`
- `artifacts/stage_h_pretrained/client/client_secret.pt`

### 两种使用形态

#### 形态 A：本地演示 / 一体化测试

使用：

- `model_state.pt`

这是一份包含模型与客户端秘密的 demo bundle，适合本地完整演示和快速回归。

#### 形态 B：client / server 分离

更贴近论文部署方式的分离形态是：

- server 持有：
  - `server_model_state.pt`
  - `metadata.json`
- client 持有：
  - `client_secret.pt`

其中 `client_secret.pt` 包含：

- `perm_vocab`
- `inv_perm_vocab`

### 输入映射与输出恢复在哪里做

当前代码里，输入 token 侧的“加密 / 混淆”在：

- `src/transforms.py`
  - `map_input_ids(...)`

输出 token/logit 侧的“解密 / 恢复”在：

- `src/transforms.py`
  - `restore_logits(...)`
  - `unmap_output_ids(...)`

当前工件推理演示脚本：

- `scripts/export_stage_h_model.py`
- `scripts/infer_stage_h_model.py`

已经验证：

- stage-H 工件可以重新加载
- 可以直接做推理测试

这意味着当前已经有：

> **一版可直接交付 server、并由 client 负责输入映射和输出恢复的 stage-H 混淆模型工件。**

#### 形态 C：pretrained-like bundle

为了更接近 `save_pretrained` 体验，当前还提供：

- `scripts/export_stage_h_pretrained.py`
- `scripts/infer_stage_h_pretrained.py`

对应目录：

- `artifacts/stage_h_pretrained/server/`
- `artifacts/stage_h_pretrained/client/`

这版仍然需要当前仓库里的自定义 loader，但目录形态已经明显更接近标准 Hugging Face 导出布局。

---

## 9. 测试情况

新增测试：

- `tests/test_stage_h.py`

已运行：

- `conda run --no-capture-output -n qwen-transformers pytest -q tests/test_stage_h.py`
- `conda run --no-capture-output -n qwen-transformers pytest -q`

结果：

- `2 passed`
- `33 passed`

---

## 10. 阶段 H 的正式结论

阶段 H 当前可以正式判定为：

### H-M1：噪声强度定标完成

- 论文默认参数点已验证
- 当前推荐工作点已确定：
  - `αe=0.1`
  - `αh=0.05`
  - `λ=0.3`
  - `h=128`
  - `β=8`
  - `γ=1e3`

### H-M2：attention 静态化收敛完成

- runtime 复杂结构明显减少
- q/k/v/o 权重承担更多混淆表达职责
- block0 / prefix-2 不劣化

### H-M3：系统联合回归完成

- prefix-4 / full-layer 保持阶段 G 同量级恢复能力
- 阶段 H 可正式判定完成

### H-M4：工件交付完成

- stage-H 工件已导出
- client/server 分离边界已明确
- server 侧与 client 侧的职责已清楚定位

因此当前项目状态可以更准确地描述为：

> **已经完成 AloePri 技术报告第 5 节方法实现的复现，并进一步把 attention 的工程表达向论文离线混淆部署形态收敛；同时完成了 embed/head 噪声强度的功能定标，但尚未进入第 6 节的隐私攻击评估。**

---

## 11. 一句话总结

> **阶段 H 已完成两件收口工作：一是锁定了当前实现下可复用的非零噪声工作点，二是把 attention 的复杂 runtime 结构继续静态化收敛到离线权重表达中，而不破坏系统恢复能力。**
