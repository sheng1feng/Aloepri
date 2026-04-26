# 阶段 J：标准形状前缀恢复报告

> Legacy note: 本文档对应的是 **旧版 Stage J**，其重点是 conservative standard-shape / square-monomial prefix 与 full-layer 恢复。新版 `Stage J` 的 canonical 定义见 `docs/阶段J_论文一致部署路线说明.md`。

本文档记录阶段 J 第一轮实现结果。

本轮目标不是部署，也不是 vLLM，而是回答：

> **在保持原始 Qwen2 标准 checkpoint shape 不变的前提下，non-expanding hidden transform 能否从 `embed/head-only` 推进到 `block0 / prefix`，并重新恢复功能正确性？**

---

## 1. 前情提要

阶段 I / Phase 2 的最小原型已经证明：

- `square monomial hidden transform` 可以保持标准 checkpoint shape；
- 但如果只做 `embed/head-only`，则功能完全不成立：
  - `avg_full_logits_max_abs_error ≈ 34.61`
  - `generated_ids_exact_match_rate = 0.0`

因此，阶段 J 的核心任务不是再调 `embed/head-only`，而是：

> **把同一套 non-expanding hidden transform 继续推进到中间层。**

---

## 2. 本轮实现内容

### 2.1 新增 square-prefix 构造入口

文件：

- `src/stage_j_block0.py`

核心能力：

- `build_stage_j_square_model(...)`
  - 允许指定 `adapted_layers`
  - 当前已经用于：
    - `prefix-1`
    - `prefix-2`
    - `prefix-4`
    - `prefix-8`
    - `full-layer`
- `build_stage_j_block0_model(...)`
  - 是 `adapted_layers=[0]` 的便捷包装

### 2.2 采用的 hidden transform

本轮继续沿用阶段 I 的 non-expanding 设计：

\[
P = \Pi \cdot S_{\pm}
\]

默认：

- `global_scale = 1.0`
- 只做 permutation + sign

### 2.3 中间层适配方式

对每个被适配的 layer，当前采用标准 shape 下的直接参数变换：

- `input_layernorm.weight`
  - 只做按 hidden permutation 的重排
- `q/k/v`
  - 输入侧吸收 hidden transform
- `o_proj`
  - 输出侧吸收 hidden transform
- `gate/up`
  - 输入侧吸收 hidden transform
- `down_proj`
  - 输出侧吸收 hidden transform

这里没有使用扩维 KeyMat，也没有引入新的非标准算子语义。

### 2.4 handoff 策略

为了先验证 `prefix-n` 的局部正确性，本轮在“最后一个已适配层”之后引入了一个 **handoff wrapper**：

- 作用：把 obfuscated hidden 恢复回 base hidden
- 这样后续未适配层可以继续按 baseline 路径运行

因此本轮结果可分成两类：

- `prefix-n + handoff`
  - 用于 `prefix-1 / 2 / 4 / 8`
- `full-layer without handoff`
  - 用于完整 24 层

也就是说，本轮已经不再只是“prefix 可行性验证”，而是已经推进到了 **full-layer 标准形状版本**。

---

## 3. 本轮排错过程

这一轮并不是一次成功，过程中定位并修复了两个关键 bug。

### 3.1 Bug 1：`o_proj / down_proj` 输出侧权重方向写反

最初 block0 回归时，现象是：

- `q/k/v` 几乎精确对齐
- 但 `attn_out / mlp_out / block_out` 明显偏离

根因：

- `o_proj` 与 `down_proj` 这种“输出回到 hidden”的线性层，左乘方向写反了

修复后：

- block0 的隐藏状态恢复明显正常

### 3.2 Bug 2：`lm_head` 与 `embed_tokens` 是 tied weights

进一步排查时发现：

- `embed_out`
- `block0_out`
- `prefix hidden_states`

已经几乎精确恢复，但 `final_logits` 仍然非常差。

根因：

- Qwen2 的 `lm_head.weight` 与 `embed_tokens.weight` 共享参数
- 改 embedding 时把 head 也一起改坏了

修复方式：

- 显式 untie `lm_head`
- 保留 Stage A 的 permuted head 作为输出头

修复后：

- `final_logits`
- `greedy`
- `generated ids/text`

全部恢复到近乎精确一致

---

## 4. 结果

### 4.1 prefix-1（block0）

结果文件：

- `outputs/stage_j/prefix_1_square.json`

汇总结果：

- `avg_final_logits_restored_max_abs_error ≈ 1.20e-4`
- `avg_layer_0_block_out_max_abs_error ≈ 1.55e-5`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

结论：

> **block0 已经在标准 shape + non-expanding transform 下恢复成立。**

### 4.2 prefix-2

结果文件：

- `outputs/stage_j/prefix_2_square.json`

汇总结果：

- `avg_final_logits_restored_max_abs_error ≈ 1.75e-4`
- `avg_layer_0_block_out_max_abs_error ≈ 1.55e-5`
- `avg_layer_1_block_out_max_abs_error ≈ 1.74e-5`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

结论：

> **prefix-2 已经成立，没有出现断崖爆炸。**

### 4.3 prefix-4

结果文件：

- `outputs/stage_j/prefix_4_square.json`

汇总结果：

- `avg_final_logits_restored_max_abs_error ≈ 1.34e-4`
- `avg_layer_2_block_out_max_abs_error ≈ 6.71e-4`
- `avg_layer_3_block_out_max_abs_error ≈ 2.44e-4`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

结论：

> **prefix-4 也已经保持稳定，且恢复能力没有明显退化。**

### 4.4 prefix-8

结果文件：

- `outputs/stage_j/prefix_8_square.json`

汇总结果：

- `avg_final_logits_restored_max_abs_error ≈ 1.91e-4`
- `avg_layer_7_block_out_max_abs_error ≈ 3.66e-4`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

结论：

> **prefix-8 仍然稳定，没有出现传播失控。**

### 4.5 full-layer

结果文件：

- `outputs/stage_j/full_layers_square.json`

汇总结果：

- `avg_final_logits_restored_max_abs_error ≈ 1.26e-4`
- `avg_layer_23_block_out_max_abs_error ≈ 5.67e-4`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

结论：

> **full-layer 已经在标准 shape 路线下恢复到近乎精确正确。**

### 4.6 full-layer 标准 checkpoint 导出回归

导出文件：

- `artifacts/stage_j_full_square/server/`
- `artifacts/stage_j_full_square/client/client_secret.pt`

回归结果：

- `outputs/stage_j/full_layers_square_export_regression.json`

汇总结果：

- `avg_full_logits_max_abs_error ≈ 1.26e-4`
- `avg_last_token_logits_max_abs_error ≈ 2.26e-5`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

这说明：

> **full-layer 标准形状版本不只是“内存里能跑”，而且已经可以导出为标准 HF checkpoint 并保持同级别恢复能力。**

### 4.7 回接阶段 I / vLLM 主线的验证

为了确认阶段 J 的 full-layer 标准 checkpoint 是否已经满足阶段 I 的“标准入口”要求，本轮还额外执行了：

- `scripts/run_stage_i_vllm_regression.py`
- `server_dir = artifacts/stage_j_full_square/server`
- `client_secret = artifacts/stage_j_full_square/client/client_secret.pt`

结果文件：

- `outputs/stage_j/full_layers_square_vllm_regression.json`

结果仍然是：

- `available = true`
- `skipped = true`
- `reason = "vllm import succeeded but runtime initialization failed: RuntimeError: Device string must not be empty"`

因此可以明确写出：

> **阶段 J 这版 full-layer 标准 checkpoint 已经满足“模型形状与功能正确性”要求；当前没有打通 vLLM 的原因仍是本机 CPU backend 环境阻塞，而不是标准形状模型本身有问题。**

---

## 5. 这一步说明了什么

阶段 J 第一轮已经回答了一个很关键的问题：

> non-expanding / shape-preserving hidden transform 并不是“只能保持 shape，功能做不回来”；只要继续把中间层按标准 shape 做协变适配，它可以恢复到近乎精确正确。

也就是说，阶段 I 里发现的“`embed/head-only` 完全失败”，并不是说明这条路线错了，而只是说明：

> **中间层不能不改。**

现在这一点已经被 `prefix-1 / 2 / 4 / 8 / full-layer` 的结果直接验证了。

---

## 6. 当前边界

这轮阶段 J 虽然已经非常成功，但要准确描述它的边界：

### 已完成

- non-expanding hidden transform
- block0 恢复
- prefix-2 恢复
- prefix-4 恢复
- prefix-8 恢复
- full-layer 恢复
- full-layer 标准 checkpoint 导出与 HF 回归

### 还没完成

- 在当前环境中完成 vLLM 实跑验证
- 解决 CPU backend / 官方 CPU wheel 获取问题

此外，本轮后续已经补做了 standard-shape full-layer 的噪声定标，详见：

- `docs/阶段J_标准形状噪声定标报告.md`

当前已经得到一组推荐非零工作点：

- `alpha_e = 0.02`
- `alpha_h = 0.01`

并导出了对应工件：

- `artifacts/stage_j_full_square_tiny_a/`

进一步地，后续又把：

- `artifacts/stage_j_full_square/`
- `artifacts/stage_j_full_square_tiny_a/`

统一收拢成了阶段 K 发布目录：

- `artifacts/stage_k_release/`

对应说明见：

- `docs/阶段K_标准形状交付包装报告.md`

也就是说，本轮更准确的状态是：

> **阶段 J 已经完成“标准形状 full-layer 恢复”的可行性验证。**

---

## 7. 下一步建议

接下来的最自然顺序是：

1. 把当前 `artifacts/stage_j_full_square/` 重新接回阶段 I 的 vLLM 主线
2. 优先验证 vLLM 是否能直接加载这版 full-layer 标准 checkpoint
3. 若 vLLM 仍因 CPU backend 环境阻塞，则先把这版模型纳入标准部署说明

换句话说：

- **阶段 I** 提供标准导出入口
- **阶段 J** 把标准形状方案真正做成功
- 然后才能重新回到部署验证

---

## 8. 一句话结论

> 阶段 J 已经证明：**non-expanding / 标准 shape 路线不仅在结构上可行，而且已经在 `prefix-1 / 2 / 4 / 8` 以及 `full-layer` 上恢复到近乎精确正确；当前剩下的核心工作是把这版 full-layer 标准 checkpoint 重新接回阶段 I 的 vLLM 主线。**
