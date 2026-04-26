# AloePri 技术报告复现总报告（阶段 A ～ E）

本文档基于技术报告  
`docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf`  
以及当前仓库已经完成的阶段 A、B、C、D、E 实验结果，给出一份**完整、详细、可追踪**的复现总报告。

报告目标有三个：

1. 说明当前复现做到了什么；
2. 给出每个阶段的实现边界、实验设计和核心数据；
3. 明确当前与论文完整版 AloePri 之间还有哪些差距。

---

## 1. 论文目标与当前复现定位

### 1.1 原论文在做什么

技术报告提出 **AloePri**，其核心思想是：

- 不只对输入数据做混淆；
- 而是对：
  - 输入 token
  - embedding / model head
  - attention
  - FFN
  - normalization
  - 输出 token
  
共同构造一个 **covariant obfuscation**（协变混淆）体系；

使得：

\[
\tilde{f}(\phi_X(x), \phi_\Theta(\theta)) \approx \phi_Y(f(x,\theta))
\]

也就是说：

- 模型在混淆空间中继续计算；
- 结果在反混淆后尽量与原始模型一致；
- 同时让服务端直接观察到的输入、权重、中间状态和输出更难恢复明文。

### 1.2 当前复现的定位

当前仓库并**没有**完整复现论文最终工业版 AloePri。  
当前完成的是一条**逐阶段、逐模块、逐层推进**的复现路线：

- **阶段 A：词表空间闭环**
- **阶段 B：block0 attention 子层恢复**
- **阶段 C：block0 完整 block 恢复**
- **阶段 D：多层 block 级恢复验证**
- **阶段 E：复杂 Attention 结构补全与修复**

这条路线的特点是：

- 先证明最小闭环正确；
- 再证明局部模块正确；
- 再证明完整 block 正确；
- 最后再证明多层传播是可控的。

这种推进方式和论文“先构造子混淆，再用组合定理拼成整体”的思路是一致的。

---

## 2. 复现环境与固定实验设置

### 2.1 模型

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 本地路径：`model/Qwen2.5-0.5B-Instruct`

### 2.2 运行环境

- Python：`3.11`
- 依赖环境文件：`environment.qwen-transformers.yml`
- 关键依赖：
  - `torch`
  - `transformers`
  - `accelerate`
  - `safetensors`
  - `pytest`

### 2.3 固定 prompt

所有阶段统一使用 5 条 prompt：

1. `请用一句话介绍你自己。`
2. `什么是 Transformer block？`
3. `请解释一下注意力机制。`
4. `Hello, how are you?`
5. `Write one sentence about machine learning.`

### 2.4 固定随机性

所有阶段统一固定：

- Python 随机种子
- NumPy 随机种子
- Torch 随机种子
- 词表 permutation 种子
- hidden permutation / scaling 种子
- FFN permutation / scaling 种子

默认种子：

- `20260323`

---

## 3. 当前代码实现总览

当前核心实现分为 7 个模块族：

### 3.1 基础工具

- `src/model_loader.py`
- `src/evaluator.py`
- `src/defaults.py`

作用：

- 加载模型与 tokenizer
- 固定种子
- 统一 prompt 编码
- 统一误差计算与 JSON 落盘

### 3.2 阶段 A：词表空间

- `src/key_manager.py`
- `src/transforms.py`
- `src/obfuscate_embed_head.py`
- `scripts/run_baseline.py`
- `scripts/run_permuted_eval.py`

### 3.3 阶段 B：hidden-space + block0 attention

- `src/hidden_keys.py`
- `src/stage_b.py`
- `scripts/run_stage_b_hidden_only.py`
- `scripts/run_stage_b_block0_attn_wrapper.py`
- `scripts/run_stage_b_block0_attn_fused.py`

### 3.4 阶段 C：block0 norm + FFN + full block

- `src/obfuscate_rmsnorm.py`
- `src/obfuscate_ffn.py`
- `src/stage_c.py`
- `scripts/run_stage_c_block0_full.py`

### 3.5 阶段 D：多层 block 级推广

- `src/stage_d.py`
- `scripts/run_stage_d_layers.py`
- `scripts/run_stage_d_layers_2.py`
- `scripts/run_stage_d_layers_4.py`
- `scripts/run_stage_d_layers_8.py`
- `scripts/run_stage_d_layers_full.py`

### 3.6 阶段 E：复杂 Attention 结构

- `src/attention_keys.py`
- `src/gqa_layout.py`
- `src/obfuscate_attention_complex.py`
- `src/stage_e.py`
- `scripts/run_stage_e_ablation.py`
- `scripts/run_stage_e_block0_attention_complex.py`
- `scripts/run_stage_e_prefix_layers.py`
- `scripts/run_stage_e_head_trace_check.py`

### 3.7 测试

- `tests/test_key_manager.py`
- `tests/test_transforms.py`
- `tests/test_obfuscate_embed_head.py`
- `tests/test_hidden_keys.py`
- `tests/test_stage_b.py`
- `tests/test_stage_c.py`
- `tests/test_stage_d.py`

当前测试总数：

- `25 passed`

---

## 4. 阶段 A：词表空间闭环复现

## 4.1 阶段 A 目标

阶段 A 只处理**词表空间**，不引入 hidden-space basis 变化。

它验证的是：

\[
\text{Model}(x)
\equiv
\text{inv\_perm}\big(\text{Model}_{perm}(\text{perm}(x))\big)
\]

也就是：

- 输入 token ids 置换；
- embedding 行置换；
- `lm_head` 词表方向置换；
- 输出 logits 再逆置换；
- 恢复后应与原模型完全一致。

## 4.2 阶段 A 实现要点

### 词表置换

实现：

- `perm_vocab`
- `inv_perm_vocab`

当前策略不是“整张 embedding 全部打乱”，而是保守策略：

- 普通词表区间可置换；
- special token 固定；
- added token 固定；
- `len(tokenizer)` 之外的 tail rows 固定。

这是保证 chat template 和 tokenizer 行为稳定的关键。

### embedding / lm_head

当前实现保证：

- 若原 token `i` 被映射到混淆 token `j = perm_vocab[i]`
- 则：
  - `embedding[j] = embedding[i]`
  - `lm_head[j]` 对应原词表语义 `i`

### 在线恢复

当前在线过程是：

1. `input_ids -> perm_vocab[input_ids]`
2. 混淆模型推理
3. `logits_perm -> restore_logits(...)`
4. 从恢复后的 logits 中选 token

## 4.3 阶段 A 校验命令

```bash
conda run --no-capture-output -n qwen-transformers pytest -q
conda run --no-capture-output -n qwen-transformers python scripts/run_baseline.py --max-new-tokens 8
conda run --no-capture-output -n qwen-transformers python scripts/run_permuted_eval.py --max-new-tokens 8
```

并额外执行严格校验，输出：

- `outputs/permuted_eval/stage_a_strict_verification.json`

## 4.4 阶段 A 核心结果

来自 `outputs/permuted_eval/stage_a_strict_verification.json`：

- `prompt_count = 5`
- 所有 prompt：
  - `full_logits_equal = true`
  - `max_abs_error = 0.0`
  - `mean_abs_error = 0.0`
  - `generated_ids_equal = true`
  - `generated_text_equal = true`

额外校验：

- `special_ids_fixed = true`
- `added_ids_fixed = true`
- `tail_rows_fixed = true`

## 4.5 阶段 A 结论

阶段 A 已经做到：

- 全位置 full logits 严格一致
- greedy 短生成严格一致
- 文本解码结果严格一致

结论：

> **阶段 A 已完成，并且是严格数学闭环，不是近似闭环。**

---

## 5. 阶段 B：hidden-space 最小 key + block0 attention 恢复

## 5.1 阶段 B 目标

阶段 B 的目标不是恢复整层 block，而是验证：

\[
\text{Attention}_0(xP_h) \approx \text{Attention}_0(x)P_h
\]

并证明：

- 只改 embed/head 而不改 attention 会失配；
- 只修 block0 attention 子层即可显著改善 `layer_0_attn_out`。

## 5.2 阶段 B hidden transform

当前采用最小 hidden transform：

\[
P_h = \Pi_h \cdot S_h
\]

其中：

- `\Pi_h`：hidden permutation
- `S_h`：diagonal scaling

这不是论文 Algorithm 1 的完整版 KeyMat，但足够用于验证 hidden-space 协变关系。

## 5.3 阶段 B 三种模式

### hidden_only

- embed/head 接 hidden transform
- attention 不修

### block0_attn_wrapper

- block0 attention 显式 wrapper 修复

### block0_attn_fused

- 把 wrapper 中的 basis 变换吸收到 `q/k/v/o` 权重

## 5.4 阶段 B 关键数据

### hidden_only

- `avg_embed_out_restored_max_abs_error = 5.215e-09`
- `avg_layer_0_attn_out_restored_max_abs_error = 0.4838`
- `avg_layer_0_block_out_restored_max_abs_error = 3.4829`
- `avg_final_logits_restored_max_abs_error = 29.6520`

说明：

- embed/head 的 hidden-space 接入方向正确；
- 但 block0 attention 明显失配。

### block0_attn_wrapper

- `avg_layer_0_attn_out_restored_max_abs_error = 0.26317`

相较 `hidden_only`：

- 从 `0.4838 -> 0.2632`

说明：

- block0 attention 修复有效。

### block0_attn_fused

- `avg_layer_0_attn_out_restored_max_abs_error = 0.263174`

与 wrapper 几乎一致，说明：

- wrapper 与 fused-weight 实现等价。

## 5.5 阶段 B 结论

阶段 B 已经严格证明：

1. hidden transform 一旦引入，不修 attention 会失配；
2. 修 block0 attention 后 `attn_out` 显著恢复；
3. fused-weight 版与 wrapper 版数值一致。

结论：

> **阶段 B 已完成，达成“从词表正确到 attention 正确”的过渡。**

---

## 6. 阶段 C：block0 完整 block 恢复

## 6.1 阶段 C 目标

阶段 C 要恢复：

\[
\text{Block}_0(xP_h) \approx \text{Block}_0(x)P_h
\]

也就是把：

- `input_layernorm`
- `self_attn`
- residual
- `post_attention_layernorm`
- `mlp`
- residual

整体恢复到协变正确。

## 6.2 阶段 C 新增机制

### RMSNorm 修正

通过：

\[
\kappa = E[\|xP_h\| / \|x\|]
\]

对 hidden transform 造成的范数偏移做补偿。

当前实现中：

- `κ` 通过采样估计；
- `κ` 作用在 RMSNorm 输出之后；
- norm weight 按 hidden permutation 对齐。

### FFN 修正

当前引入：

- `Z_ffn`：中间维 permutation
- `H_ffn`：中间维 scaling

并要求：

- `gate_proj` 与 `up_proj` 共享同一组 `Z_ffn`

这是 FFN 能保持非线性结构正确的必要条件。

## 6.3 阶段 C 对照组

### hidden_only

- attention 不修
- norm 不修
- ffn 不修

### attn_only

- attention 修
- norm 不修
- ffn 不修

### full_block

- attention 修
- 两个 RMSNorm 都修
- FFN 修

## 6.4 阶段 C 关键数据

当前 `kappa = 1.0005348920822144`

| 模式 | input_norm_out | attn_out | post_attn_norm_out | mlp_out | block_out | final_logits |
|---|---:|---:|---:|---:|---:|---:|
| hidden_only | 2.2462 | 0.4838 | 17.0623 | 3.4959 | 3.4829 | 29.6520 |
| attn_only | 2.2462 | 0.2632 | 14.0347 | 3.1655 | 3.1904 | 33.1681 |
| full_block | 0.00519 | 0.000682 | 0.11896 | 0.04710 | 0.04713 | 35.5028 |

## 6.5 阶段 C 解读

### 成功的部分

- `input_norm_out`：`2.2462 -> 0.00519`
- `attn_out`：`0.2632 -> 0.000682`
- `post_attn_norm_out`：`14.0347 -> 0.11896`
- `mlp_out`：`3.1655 -> 0.04710`
- `block_out`：`3.1904 -> 0.04713`

说明：

> block0 完整内部计算已经显著恢复。

### 还没成功的部分

- `final_logits` 没回归，反而略升高。

这不表示阶段 C 失败，因为：

- block0 修好了；
- 后续 23 层没修；
- block0 输出继续流入仍失配的后续网络。

## 6.6 阶段 C 结论

阶段 C 已达成：

- RMSNorm 协变修正成立
- FFN 协变 wrapper 成立
- block0 完整恢复成立

结论：

> **阶段 C 已完成，达成“从 attention 正确到 block 正确”的过渡。**

---

## 7. 阶段 D：多层 block 级推广验证

## 7.1 阶段 D 目标

阶段 D 的目标是把 block0 的恢复逻辑推广到更多层，并验证：

\[
\text{Stack}_{0:k}(xP_h) \approx \text{Stack}_{0:k}(x)P_h
\]

也就是不再只看单层，而是看**前 k 层组合**在 obfuscated basis 下是否仍近似等价。

阶段 D 要回答三件事：

1. block0 方案能否无修改复制到 block1、block2...；
2. 多层组合后误差传播是可控还是爆炸；
3. 随修复层数增加，`final_logits` 与 greedy 生成是否开始系统回归。

## 7.2 阶段 D 新增机制

### 多层 hook 抽象

当前新增：

- 任意 `layer_idx` 的 attention / norm / FFN wrapper 挂载；
- 任意 `layer_idx` 的 trace 记录。

### per-layer / per-norm κ

不再全层共享一个 `κ`，而是对每层的：

- `input_layernorm`
- `post_attention_layernorm`

分别估计 `κ`。

这是阶段 D 相较阶段 C 的关键改进之一。

### 手动 greedy 生成

阶段 D 中，为避免“推理链路被内部 generate 缓存逻辑掩盖”，加入了手动 greedy token 生成，用恢复后的 logits 逐 token 推进。

## 7.3 阶段 D 实验规模

共跑了 4 组前缀层实验：

- `layers_2`
- `layers_4`
- `layers_8`
- `layers_full`

每组都比较三种模式：

- `hidden_only`
- `block0_only`
- `prefix_full`

其中 `prefix_full` 表示：

- 对前缀层 `[0..k]` 全部应用 attention + norm + ffn 恢复；
- 对其后层保持原 hidden-only 状态。

---

## 8. 阶段 D 关键数据

## 8.1 两层实验（layers_2）

### final logits

- `hidden_only`：`29.6520`
- `block0_only`：`35.3178`
- `prefix_full@[0,1]`：`31.2628`

### `layer_1_block_out`

- `hidden_only`：`6.9973`
- `block0_only`：`4.3486`
- `prefix_full`：`0.03069`

解释：

- 说明 layer1 也能像 layer0 一样被恢复；
- 且误差没有在第二层爆炸。

## 8.2 四层实验（layers_4）

### `layer_3_block_out`

- `hidden_only`：`1673.6501`
- `block0_only`：`1672.6854`
- `prefix_full`：`29.9940`

解释：

- 如果只修 block0，误差在第 3 层已经爆炸；
- 若对前 4 层都恢复，误差被压回到了可控范围。

## 8.3 八层实验（layers_8）

### `layer_7_block_out`

- `hidden_only`：`1701.8278`
- `block0_only`：`1701.2686`
- `prefix_full`：`30.8612`

解释：

- 误差增长仍存在；
- 但前 8 层全部恢复后，误差不再是无控制地爆炸。

## 8.4 全层实验（layers_full）

### final logits

- `hidden_only`：`29.6520`
- `block0_only`：`35.3178`
- `prefix_full`：`20.0770`

### 生成行为

- `greedy_first_token_match_rate = 0.8`
- `generated_ids_exact_match_rate = 0.4`
- `generated_text_exact_match_rate = 0.4`

### `layer_23_block_out`

- `hidden_only`：`68.2709`
- `block0_only`：`125.9776`
- `prefix_full`：`7.9754`

解释：

- 当恢复推广到全 24 层时，最终 logits 开始**系统回归**；
- greedy 首 token 和部分生成文本也开始恢复；
- 这说明阶段 D 已经开始触达系统级改进，而不只是局部 block 改进。

---

## 9. 阶段 D 逐 prompt 现象（全层）

全层实验中 `prefix_full` 的逐 prompt 结果呈现出清晰的恢复趋势：

### Prompt 1：请用一句话介绍你自己。

- baseline：`我是一个人工智能助手，可以帮助您解答`
- `prefix_full`：`我是来自阿里云的大规模语言模型`

解释：

- 文本未完全一致；
- 但已经从乱码/无意义 token 恢复到了语义相关、流畅的自然语言句子。

### Prompt 2：什么是 Transformer block？

- baseline：`Transformer block 是一种特殊的神经网络结构`
- `prefix_full`：`Transformer Blocks 是Transformer模型的一部分，它是`

解释：

- 首 token 已匹配；
- 文本开始恢复到主题相关生成。

### Prompt 3：请解释一下注意力机制。

- baseline：`注意力机制是一种神经网络技术，用于`
- `prefix_full`：`注意力机制是一种神经网络模型，它`

解释：

- 已明显恢复到正确主题；
- 虽然不完全一致，但不再是乱码。

### Prompt 4：Hello, how are you?

- baseline：`Hello! I'm an artificial intelligence language`
- `prefix_full`：`Hello! I'm an artificial intelligence language`

解释：

- 完全一致。

### Prompt 5：Write one sentence about machine learning.

- baseline：`Machine learning allows computers to learn from data`
- `prefix_full`：`Machine learning allows computers to learn from data`

解释：

- 完全一致。

## 9.1 阶段 D 总体解读

阶段 D 的结果可以概括成三点：

1. block 级恢复方案可以稳定复制到更多层；
2. 多层传播中的误差虽然仍增长，但已从“无控制爆炸”变成“可控增长”；
3. 当恢复推广到全层时，最终 logits 和生成行为开始明显回归。

因此，阶段 D 达成了它的核心目标：

> **验证了多层 block 级协变恢复是可行的。**

---

## 10. 阶段 E：复杂 Attention 结构补全与修复

## 10.1 阶段 E 目标

阶段 E 对应论文第 `5.2.3` 节，目标是把 attention 从：

- 阶段 D 的 `simplified attention`

升级为更接近论文结构的版本，逐步补齐：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`
- `τ_kv`
- `τ_group`

阶段 E 要验证两件事：

1. 复杂 attention 结构接入后，是否仍能保持当前 block 恢复链路；
2. `τ_kv / τ_group` 是否真正改变了内部 head-level 结构，而不是形同虚设。

## 10.2 阶段 E 新增模块

### `src/attention_keys.py`

负责生成：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`
- `τ_kv`
- `τ_group`

并支持 profile：

- `simplified`
- `rqk`
- `rqk_hqk`
- `rqk_hqk_block`
- `rqk_hqk_block_taukv`
- `rqk_hqk_block_taukv_taugroup`

### `src/gqa_layout.py`

负责 Qwen GQA 头布局：

- query heads
- kv heads
- groups

支持：

- grouped query reshape / merge
- `τ_kv`
- `τ_group`

### `src/obfuscate_attention_complex.py`

负责：

- 复杂 attention wrapper
- raw head-level trace
- intra-head Q/K 结构融合

### `src/stage_e.py`

负责把复杂 attention 结构接入现有：

- RMSNorm wrapper
- FFN wrapper
- layerwise hook 框架

## 10.3 阶段 E 最关键的排错与修复

### 第一个精确定位

修复前做了一个直接的 attention score 检查，发现：

- `rqk` score max error ≈ `412`
- `rqk_hqk` ≈ `409`
- `rqk_hqk_block` ≈ `337`

这说明问题并不在 residual 或 FFN，而是：

> **Q/K 结构在进入 softmax 之前就已经坏了。**

### 根因

最终定位到：

- `R̂_qk / Ĥ_qk / Ẑ_block` 的维度配对方式
- 与 Qwen2 的 `rotate_half` 语义不一致

修复前按“相邻 2x2 block”组织；  
修复后改成与 Qwen2 实际 RoPE 一致的：

- 前半维 / 后半维配对

对应修复点：

- `src/attention_keys.py`

修复后重新验证：

- `rqk` score max error：`0.0048828125`
- `rqk_hqk`：`0.0048828125`
- `rqk_hqk_block`：`0.0048828125`

结果文件：

- `outputs/stage_e/diagnostics_after_fix.json`

### 第二个工程修复

在 `τ_kv / τ_group` 的排查中发现：

- 当 head 数量很小（比如 `num_kv_heads = 2`）时，随机采样 permutation 可能正好得到 identity；
- 这会让实验误判“置换无效”。

因此对：

- `generate_tau_kv(...)`
- `generate_tau_group(...)`

加入了防恒等排列修正：

- 若采样到 identity，则重采样；
- 若仍为 identity，则退化为循环移位。

## 10.4 阶段 E 单层 block0 结果

来自：

- `outputs/stage_e/block0_attention_complex.json`

修复后的关键结果如下：

| Profile | `q_proj_out` | `k_proj_out` | `v_proj_out` | `attn_out` | `block_out` | `final_logits` |
|---|---:|---:|---:|---:|---:|---:|
| `simplified` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk` | 0.019803 | 0.010511 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk_block` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk_block_taukv` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |
| `rqk_hqk_block_taukv_taugroup` | 0.019803 | 0.010513 | 0.000426 | 0.000529 | 0.024588 | 35.3178 |

这说明：

> 修复后的复杂 attention 结构版已经在 block0 单层上完全回到阶段 D simplified 基线。

## 10.5 `τ_kv / τ_group` 的内部作用

来自：

- `outputs/stage_e/head_trace_check.json`

在恢复后的功能指标上：

- `rqk_hqk_block`
- `rqk_hqk_block_taukv`
- `rqk_hqk_block_taukv_taugroup`

表现保持中性。

但在未恢复前的 raw head-level trace 上，已经明确观察到非平凡差异：

### `rqk_hqk_block` vs `rqk_hqk_block_taukv`

- `q_heads_post_inter_raw = 55.9384`
- `k_heads_post_inter_raw = 272.7900`
- `v_heads_post_inter_raw = 0.3084`
- `attn_heads_pre_inverse_raw = 0.3084`

### `rqk_hqk_block_taukv` vs `rqk_hqk_block_taukv_taugroup`

- `q_heads_post_inter_raw = 89.9669`
- `attn_heads_pre_inverse_raw = 0.4094`

这说明：

> `τ_kv / τ_group` 并不是 no-op；它们确实改变了恢复前的内部排列结构，只是恢复后的功能指标按设计保持不变。

## 10.6 阶段 E prefix 多层结果

### prefix-2

- `outputs/stage_e/prefix_layers_2.json`

关键结果：

- `layer_0_block_out = 0.024588`
- `layer_1_block_out = 0.030693`
- `final_logits = 31.2628`

### prefix-4

- `outputs/stage_e/prefix_layers_4.json`

关键结果：

- `layer_2_block_out = 3.1821`
- `layer_3_block_out = 29.9940`
- `final_logits = 31.7846`

### prefix-8

- `outputs/stage_e/prefix_layers_8.json`

关键结果：

- `layer_7_block_out = 30.8612`
- `final_logits = 31.9148`

### prefix-full

- `outputs/stage_e/prefix_layers_full.json`

关键结果：

- `layer_23_block_out = 7.9754`
- `avg_final_logits_restored_max_abs_error = 20.0770`
- `greedy_first_token_match_rate = 0.8`
- `generated_text_exact_match_rate = 0.4`

这些结果已经重新与阶段 D 的系统级结果对齐。

## 10.7 阶段 E 结论

阶段 E 现在可以明确判定为：

1. 论文 attention 五类复杂结构已经全部接入；
2. 核心功能 bug 已定位并修复；
3. 复杂 attention 结构版已经回到阶段 D 的稳定功能基线；
4. `τ_kv / τ_group` 的内部作用也已被显式验证。

因此：

> **阶段 E 已完成从“复杂 attention 结构接入”到“复杂 attention 结构可稳定工作”的过渡。**

---

## 11. 当前复现与论文完整版的差异

尽管已经做到阶段 D，但当前实现仍然不是论文完整版 AloePri。

### 当前已经实现的

- 词表置换闭环
- hidden-space 简化 key（permutation + scaling）
- attention 基础协变恢复
- RMSNorm `κ` 补偿
- FFN 中间维 permutation + scaling
- block0 完整恢复
- 多层 block 级恢复验证

### 当前尚未实现的

- 论文 Algorithm 1 的 `KeyMatGen / InvKeyMatGen`
- embedding/head 的噪声项
- attention 的 `R̂_qk`
- attention 的 `Ĥ_qk`
- `Ẑ_block` 动态 block permutation
- `τ_kv`
- `τ_group`
- 更贴近论文攻击面和安全评估的实验

因此，当前工作的准确定位应是：

> **简化 AloePri 路线下的结构正确性复现与多层传播验证**

而不是：

> **完整工业版 AloePri 复现**

---

## 12. 阶段 A～E 的总体结论

### 阶段 A

达成：

- **词表正确**

结论：

- full logits、greedy、生成文本都严格闭环。

### 阶段 B

达成：

- **attention 正确**

结论：

- hidden transform 一旦引入，不修 attention 会失配；
- 修 block0 attention 后 `attn_out` 明显恢复。

### 阶段 C

达成：

- **block 正确**

结论：

- block0 的 norm + attention + ffn 全部恢复；
- `layer_0_block_out_restored ≈ baseline`

### 阶段 D

达成：

- **多层 block 级传播开始成立**

结论：

- 多层 block 恢复可以复制；
- 误差传播开始可控；
- 全层实验里 final logits 和生成开始回归。

### 阶段 E

达成：

- **复杂 attention 结构正确接入并恢复到稳定基线**

结论：

- `R̂_qk / Ĥ_qk / Ẑ_block / τ_kv / τ_group` 已全部接入；
- 关键 RoPE pairing bug 已修复；
- block0、prefix 多层、全层结果重新回到阶段 D 的稳定功能链路；
- `τ_kv / τ_group` 的内部作用已被显式验证。

---

## 13. 当前正式判定

基于当前全部实验和数据，可以给出如下正式判定：

> **当前仓库已经完成 AloePri 技术报告的一条简化复现主线：从词表闭环，到 attention 子层恢复，到 block 完整恢复，再到多层 block 级传播验证，并进一步完成了论文 attention 复杂结构的接入与修复。**

更精确地说：

- 阶段 A：严格完成
- 阶段 B：严格完成
- 阶段 C：严格完成
- 阶段 D：完成多层传播验证，并首次观察到系统级回归
- 阶段 E：完成复杂 attention 结构接入、核心 bug 修复与功能回归

---

## 14. 下一步建议

当前最合理的下一步不再是继续修 attention 功能，而是转向：

- 更细粒度的 head-level / attention-map 内部诊断，用于安全性与混淆强度分析；
- 或进入更贴近论文完整版的 key 体系（Algorithm 1）。

建议顺序：

### 阶段 F

引入论文 Algorithm 1：

- `KeyMatGen`
- `InvKeyMatGen`
- `λ` 控制矩阵范数

这是从“简化 hidden transform”走向“论文原生 key 体系”的关键一步。

### 阶段 G

做更细粒度的结构级与安全性诊断：

- attention score map
- attention probability map
- grouped query / kv head layout
- 未恢复前后的 head-level trace 对比

这一步将把阶段 E 从“功能正确”推进到“内部结构可解释”。

### 阶段 H

补安全性与攻击评估：

- VMA
- IA
- IMA
- ISA
- token frequency 相关攻击

---

## 14. 结语

当前复现路线的最大价值不在于“已经完整复现论文全部工程细节”，而在于：

- 它已经把论文里最难调试的一条主线——**协变恢复链路**——从最小闭环一直推进到了多层传播；
- 每一步都有实验与数据支撑，不是只靠理论推断；
- 并且已经明确区分了：
  - 当前是简化版什么；
  - 还缺论文完整版什么；
  - 下一步最值得补什么。

因此，这份复现工作已经具备很强的研究与工程价值：  
它不仅验证了论文思路在小模型上的可行性，还为后续补齐复杂 attention 结构和真正的 KeyMat 体系提供了一个稳定、可解释、可迭代的基线。
