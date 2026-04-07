# AloePri 技术报告复现总报告（阶段 A ～ K）

本文档基于技术报告  
`docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf`  
以及当前仓库已经完成的阶段 A、B、C、D、E、F、G、H、I、J、K 实验结果，给出一份**完整、详细、可追踪**的复现总报告。

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
- **阶段 F：Algorithm 1 / KeyMat 原生体系接入**
- **阶段 G：KeyMat 融合化 / 去 bridge 化**
- **阶段 H：Attention 静态化收敛与噪声强度定标**
- **阶段 I：标准 HF/vLLM 入口打通与 Phase 2 阻塞定位**
- **阶段 J：standard-shape full-layer 协变恢复**
- **阶段 K：standard-shape 工件统一交付包装**

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

当前核心实现分为 10 个模块族：

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
- `tests/test_attention_keys.py`
- `tests/test_gqa_layout.py`
- `tests/test_stage_e.py`
- `tests/test_keymat.py`
- `tests/test_keymat_embed_head.py`
- `tests/test_stage_f_block0.py`
- `tests/test_stage_g.py`

### 3.8 阶段 F：KeyMat 原生体系

- `src/keymat.py`
- `src/keymat_embed_head.py`
- `src/keymat_norm.py`
- `src/keymat_ffn.py`
- `src/keymat_attention_bridge.py`
- `src/stage_f.py`
- `scripts/run_stage_f_keymat_unit.py`
- `scripts/run_stage_f_embed_head.py`
- `scripts/run_stage_f_block0.py`
- `scripts/run_stage_f_prefix_layers.py`
- `scripts/run_stage_f_full_layers.py`

### 3.9 阶段 G：KeyMat 融合化

- `src/stage_g_norm.py`
- `src/stage_g_ffn.py`
- `src/stage_g_attention.py`
- `src/stage_g.py`
- `src/stage_g_artifact.py`
- `scripts/run_stage_g_regression.py`
- `scripts/run_stage_g_norm_block0.py`
- `scripts/run_stage_g_ffn_block0.py`
- `scripts/run_stage_g_attention_block0.py`
- `scripts/export_stage_g_model.py`
- `scripts/infer_stage_g_model.py`

### 3.10 阶段 H：Attention 静态化与噪声定标

- `src/stage_h_noise.py`
- `src/stage_h_attention_static.py`
- `src/stage_h.py`
- `src/stage_h_artifact.py`
- `scripts/run_stage_h_noise_calibration.py`
- `scripts/run_stage_h_attention_static.py`
- `scripts/run_stage_h_joint_regression.py`
- `scripts/export_stage_h_model.py`
- `scripts/infer_stage_h_model.py`

当前测试总数：

- `33 passed`

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

## 11. 阶段 F：Algorithm 1 / KeyMat 原生体系接入

## 11.1 阶段 F 目标

阶段 F 的目标不是继续扩层，也不是继续调复杂 attention，而是把阶段 A–E 中使用的：

- 简化 hidden transform
- 显式 permutation / scaling hidden basis

升级为论文 5.2.1 / 5.2.2 所要求的 **Algorithm 1 原生 KeyMat 体系**。

阶段 F 要验证的核心命题是：

1. 论文 5.2.1 的 `INIT / KeyMatGen / InvKeyMatGen` 能否稳定生成互逆的 `P / Q`；
2. 论文 5.2.2 的 embedding / head 路径，是否能在现有主线上接入；
3. block0 与 prefix 多层下，KeyMat 是否仍能保持功能正确性；
4. full-layer 下，系统行为是否仍保持稳定回归。

需要特别说明的是，当前阶段 F 的实现策略是：

> **先用 projection / lift exact bridge 显式接入 KeyMat，再讨论 fully-fused 的最终部署形态。**

因此阶段 F 的最强结论是：

- KeyMat 已**功能性接入成功**
- 但当前仍不是完全 server-only 的纯权重混淆终态

## 11.2 阶段 F 新增模块

### `src/keymat.py`

负责实现论文 5.2.1 的 Algorithm 1：

- `INIT(d, h, λ)`
- `KeyMatGen(B, E, F, Z)`
- `InvKeyMatGen(B^{-1}, E, F, Z)`

当前实现了：

- `KeyMatBases`
- `KeyMatTransform`
- `init_keymat_bases(...)`
- `generate_keymat(...)`
- `generate_inv_keymat(...)`
- `sample_null_columns(...)`
- `sample_null_rows(...)`
- `check_keymat_inverse(...)`

### `src/keymat_embed_head.py`

负责 embedding / head 路径：

- 噪声注入
- KeyMat embed/head obfuscation
- `KeyMatEmbeddingWrapper`
- `KeyMatHeadWrapper`

### `src/keymat_norm.py`

负责 KeyMat 下的 norm bridge：

- `estimate_kappa_for_keymat(...)`
- `KeyMatRMSNormBridge`
- `build_keymat_rmsnorm_wrapper(...)`

### `src/keymat_ffn.py`

负责 KeyMat 下的 FFN bridge：

- `KeyMatFFNBridge`
- `build_keymat_ffn_wrapper(...)`

### `src/keymat_attention_bridge.py`

负责把阶段 E 已稳定的 attention profile 桥接到 KeyMat hidden basis：

- `KeyMatAttentionBridge`

### `src/stage_f.py`

负责统一阶段 F：

- layer config
- per-layer κ
- prefix/full-layer 运行
- 指标汇总

### 新增脚本

- `scripts/run_stage_f_keymat_unit.py`
- `scripts/run_stage_f_embed_head.py`
- `scripts/run_stage_f_block0.py`
- `scripts/run_stage_f_prefix_layers.py`
- `scripts/run_stage_f_full_layers.py`

## 11.3 阶段 F 最关键的工程设计

阶段 F 当前没有一上来做 fully-fused 权重混淆，而是采用了显式 bridge：

1. 输入 hidden 先在 obfuscated space 中表示；
2. 模块内部通过 `Q` 投影回 base hidden；
3. 在 base hidden 上复用已经验证稳定的：
   - norm
   - attention
   - FFN
4. 模块输出再通过 `P` 抬升回 obfuscated hidden。

对应到不同模块：

- norm：`KeyMatRMSNormBridge`
- attention：`KeyMatAttentionBridge`
- FFN：`KeyMatFFNBridge`

这样做的目的非常明确：

- 先验证 Algorithm 1 的 KeyMat 在现有 A–E 主线上能否**功能正确**
- 避免在同一轮里同时重写：
  - KeyMat
  - complex attention
  - FFN
  - norm
  - residual

这也是阶段 F 能快速稳定落地的关键原因。

### 一个重要修复

阶段 F 接入时还发现了一个隐蔽工程问题：

- baseline 模型上先挂的 hooks
- 会通过 `prepare_stage_a_model -> deepcopy`
- 被复制到 stage-F model 上

从而污染 baseline recorder。

这个问题已经通过 `src/stage_f.py` 中的 `_clear_copied_hooks(...)` 修复。

## 11.4 阶段 F 的真实技术边界

这部分必须单独说明，否则很容易把阶段 F 当前结果误读成：

> “整模型 fully-obfuscated weights 已经完成”

实际上，阶段 F 当前是一个**混合实现**：

### 已经属于“真实混淆权重”的部分

- `embedding`
- `lm_head`

原因是这两部分已经直接构造了新的 obfuscated weights：

- embedding：`W_embed_obf = W_embed_noisy @ P`
- head：`W_head_obf = W_head_noisy @ Q^T`

并且在模型中已经被真正替换成新的 wrapper 模块运行。

### 仍然主要依赖显式 `Q/P` bridge 的部分

- `RMSNorm`
- `attention`
- `FFN`
- prefix/full-layer 的 layer handoff

这些模块当前的主逻辑仍然是：

1. `obfuscated hidden -> base hidden`（乘 `Q`）
2. 在 base space 上运行已验证稳定的模块
3. `base hidden -> obfuscated hidden`（乘 `P`）

因此它们当前更准确的定位是：

> **KeyMat bridge / exact bridge 版**

而不是论文最终 fully-fused 的 server-side 纯权重混淆形态。

### 为什么阶段 F 会显著优于阶段 D / E

这不是因为：

> “KeyMat 天然比简化 hidden transform 更容易”

更准确的原因是：

> 当前阶段 F 在大量模块里使用了显式的 `Q -> plain/stable module -> P` exact bridge。

因此阶段 F 当前最准确的结论应当是：

> **Algorithm 1 / KeyMat 已完成首次系统功能接入；其中 embedding/head 已是真实混淆权重，而 norm/attention/FFN 仍主要通过显式 `Q/P` bridge 保证功能正确性。**

## 11.5 KeyMat 单元实验

结果文件：

- `outputs/stage_f/keymat_unit.json`

本实验覆盖：

- `hidden_size ∈ {128, 896}`
- `h ∈ {32, 64, 128}`
- `λ ∈ {0.1, 0.3, 1.0}`

共 12 组组合。

### 代表结果

示例：

- `d=128, h=32, λ=0.1`
- `max_abs(PQ-I) = 2.2566e-07`
- `mean_abs(PQ-I) = 3.4832e-08`
- `condition_number(P) = 17.1788`

最坏 `PQ-I` 误差：

- `d=896, h=128, λ=1.0`
- `max_abs(PQ-I) = 1.9067e-06`

最大 condition number：

- `d=896, h=32, λ=1.0`
- `condition_number(P) = 771.8164`

### 结论

所有实验组合都满足：

- `passes_tolerance = true`

因此：

> **阶段 F 的数学验收已经通过：Algorithm 1 生成的 `P / Q` 在数值上稳定互逆。**

## 11.6 embed/head-only 结果

结果文件：

- `outputs/stage_f/embed_head_eval.json`

测试 3 组噪声设置：

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

### 结论

embed/head-only 的结论非常清晰：

1. `alpha = 0` 时，KeyMat embed/head 路径已经几乎退化为阶段 A 的严格功能基线；
2. 小噪声下，功能不受明显影响；
3. 中等噪声下开始出现可观测误差，但还没有“马上乱码”。

## 11.7 block0 结果

结果文件：

- `outputs/stage_f/block0_eval.json`

### `embed_head_only`

- `avg_final_logits_restored_max_abs_error = 0.0003696`
- `avg_layer_0_block_out_max_abs_error = 1.4782e-05`

### `block0_full`

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
- `generated_text_exact_match_rate = 1.0`

### 结论

block0 已经不只是“跑通”，而是：

> **在 KeyMat bridge 形式下，block0 的 norm / qkv / attn / mlp / block_out 都回到了数值误差级别。**

## 11.8 prefix-2 / prefix-4 / full-layer 结果

### prefix-2

结果文件：

- `outputs/stage_f/prefix_layers_2.json`

关键结果：

- `avg_final_logits_restored_max_abs_error = 0.0005118`
- `avg_layer_0_block_out_max_abs_error = 5.5075e-05`
- `avg_layer_1_block_out_max_abs_error = 1.0574e-04`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### prefix-4

结果文件：

- `outputs/stage_f/prefix_layers_4.json`

关键结果：

- `avg_final_logits_restored_max_abs_error = 0.002122`
- `avg_layer_2_block_out_max_abs_error = 0.006409`
- `avg_layer_3_block_out_max_abs_error = 0.010742`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### full-layer

结果文件：

- `outputs/stage_f/full_layers.json`

关键结果：

- `avg_final_logits_restored_max_abs_error = 0.003606`
- `avg_layer_23_block_out_max_abs_error = 0.003561`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

### 与阶段 D / E 的系统级比较

参考：

- `outputs/stage_f/regression_compare.json`

| 阶段 | `final_logits max error` | `layer_23 block_out max error` | greedy 首 token | generated ids |
|---|---:|---:|---:|---:|
| Stage D full | `20.0770` | `7.9754` | `0.8` | `0.4` |
| Stage E full | `20.0770` | `7.9754` | `0.8` | `0.4` |
| Stage F full | `0.003606` | `0.003561` | `1.0` | `1.0` |

### 结论

从“功能正确性”这个标准看，阶段 F 的显式 KeyMat bridge 结果已经明显强于阶段 D / E。

## 11.9 阶段 F 结论

阶段 F 目前可以明确判定为：

1. 论文 5.2.1 的 Algorithm 1 已独立实现；
2. `P @ Q ≈ I` 在多组 `(d, h, λ)` 下稳定成立；
3. 论文 5.2.2 的 embedding / head 路径已经接入现有主线；
4. block0 层面，KeyMat 下的 norm + attention + FFN 已经打通；
5. prefix 多层传播没有出现不可控爆炸；
6. full-layer 下，final logits 与生成已经回归到几乎精确一致。

但要非常准确地表述：

> 当前阶段 F 完成的是 **Algorithm 1 / KeyMat 的首次系统功能接入**。

并不是说已经完成了：

- fully-fused 的 server-only 权重混淆
- 论文最终部署形式的全部工程细节

---

## 12. 阶段 G：KeyMat 融合化 / 去 bridge 化

## 12.1 阶段 G 目标

阶段 F 已经证明：

- KeyMat 原生矩阵生成正确；
- embed/head 已是真实混淆权重；
- norm / attention / FFN 在显式 bridge 下可以稳定工作。

但阶段 F 仍然有一个关键边界：

- 内部层大多还依赖 runtime `Q/P` bridge

阶段 G 的目标就是把这部分：

\[
y \rightarrow Q \rightarrow \text{plain / stable module} \rightarrow P \rightarrow y'
\]

推进成参数级融合机制，也就是：

- G1：RMSNorm fused
- G2：FFN fused
- G3：Attention fused
- G4：系统级回归

## 12.2 阶段 G 新增模块

- `src/stage_g_norm.py`
- `src/stage_g_ffn.py`
- `src/stage_g_attention.py`
- `src/stage_g.py`
- `src/stage_g_artifact.py`

新增脚本：

- `scripts/run_stage_g_regression.py`
- `scripts/run_stage_g_norm_block0.py`
- `scripts/run_stage_g_ffn_block0.py`
- `scripts/run_stage_g_attention_block0.py`
- `scripts/export_stage_g_model.py`
- `scripts/infer_stage_g_model.py`

## 12.3 阶段 G 关键排错

阶段 G 最大的排错点发生在 G1：

### 初版失败

第一版 fused norm 仅使用标量 `κ` 来补偿 KeyMat 扭曲，结果 block0 明显失稳：

- `avg_layer_0_input_norm_out_max_abs_error ≈ 1.0764`
- `avg_final_logits_restored_max_abs_error ≈ 8.0871`
- `generated_ids_exact_match_rate = 0.0`

### 根因

对当前 Algorithm 1 生成的 KeyMat 来说：

- 单个标量 `κ` 不足以刻画输入范数的各向异性扭曲

### 修正

最终改成了基于 `QQ^T` 的二次型 fused norm：

- 在 obfuscated space 内直接用预计算 metric 恢复 base RMS
- 去掉完整 `Q -> norm -> P`
- 保留 `Wnorm` 融入后续线性层

这一步是阶段 G 能成功继续往下推进的关键。

## 12.4 G1：RMSNorm fused

结果文件：

- `outputs/stage_g/norm_block0.json`

### bridge baseline

- `avg_final_logits_restored_max_abs_error = 0.0004432`
- `avg_layer_0_block_out_max_abs_error = 5.5075e-05`
- `generated_ids_exact_match_rate = 1.0`

### fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0007758`
- `avg_layer_0_input_norm_out_max_abs_error = 2.8229e-05`
- `avg_layer_0_block_out_max_abs_error = 1.9526e-04`
- `generated_ids_exact_match_rate = 1.0`

### 结论

G1 已经完成：

> **RMSNorm 不再依赖显式 `Q -> norm -> P` 才能工作。**

## 12.5 G2：FFN fused

结果文件：

- `outputs/stage_g/ffn_block0.json`
- `outputs/stage_g/ffn_prefix_2.json`

### block0 fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0007796`
- `avg_layer_0_mlp_out_max_abs_error = 1.9631e-04`
- `avg_layer_0_block_out_max_abs_error = 1.9617e-04`
- `generated_ids_exact_match_rate = 1.0`

### prefix-2 fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0009170`
- `avg_layer_1_block_out_max_abs_error = 2.7847e-04`
- `generated_ids_exact_match_rate = 1.0`

### 结论

G2 已经完成：

> **FFN 已从 bridge 版推进到 fused 版，且 prefix-2 传播仍稳定。**

## 12.6 G3：Attention fused

结果文件：

- `outputs/stage_g/attention_block0.json`
- `outputs/stage_g/prefix_layers_2.json`

### block0 fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0007110`
- `avg_layer_0_q_proj_out_max_abs_error = 2.4176e-04`
- `avg_layer_0_attn_out_max_abs_error = 6.3062e-06`
- `avg_layer_0_block_out_max_abs_error = 1.7185e-04`
- `generated_ids_exact_match_rate = 1.0`

### prefix-2 fused candidate

- `avg_final_logits_restored_max_abs_error = 0.0008004`
- `avg_layer_1_block_out_max_abs_error = 3.1602e-04`
- `generated_ids_exact_match_rate = 1.0`

### 结论

G3 已经完成：

> **Attention 也不再依赖阶段 F 外层 KeyMat bridge 才能保持正确性。**

## 12.7 G4：系统级回归

结果文件：

- `outputs/stage_g/prefix_layers_4.json`
- `outputs/stage_g/full_layers.json`
- `outputs/stage_g/regression_compare.json`

### prefix-4 fused candidate

- `avg_final_logits_restored_max_abs_error = 0.001789`
- `avg_layer_2_block_out_max_abs_error = 0.009033`
- `avg_layer_3_block_out_max_abs_error = 0.012451`
- `generated_ids_exact_match_rate = 1.0`

### full-layer fused candidate

- `avg_final_logits_restored_max_abs_error = 0.003537`
- `avg_layer_23_block_out_max_abs_error = 0.003760`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

与阶段 F full-layer bridge 基线相比：

- bridge：`avg_final_logits_restored_max_abs_error = 0.003606`
- fused：`avg_final_logits_restored_max_abs_error = 0.003537`

说明：

> **去 bridge 化后，系统整体并没有退化，full-layer 仍保持阶段 F 同级别恢复能力。**

## 12.8 混淆模型导出与演示

阶段 G 已额外导出一版可复现、可加载、可直接演示推理的混淆模型工件：

- `artifacts/stage_g_full_obfuscated/metadata.json`
- `artifacts/stage_g_full_obfuscated/model_state.pt`

这份导出件当前大小约：

- `3.4G`

注意它的技术形态是：

- 保存的是 **fused stage-G 模型的全部 buffer 快照**
- 配合本地 baseline 架构重建器加载

也就是说，它不是 Hugging Face `save_pretrained` 意义上的通用模型目录，而是：

> **当前仓库下可重建、可运行、可直接演示的 stage-G 混淆模型工件。**

演示命令：

- 导出：`python scripts/export_stage_g_model.py`
- 推理：`python scripts/infer_stage_g_model.py --prompt "请用一句话介绍你自己。"`

## 12.9 阶段 G 结论

阶段 G 当前可以明确判定为：

1. G1：已完成 fused RMSNorm
2. G2：已完成 fused FFN
3. G3：已完成 fused Attention
4. G4：已完成 prefix / full-layer 系统级回归
5. 已导出一版可直接演示推理的 stage-G 混淆模型工件

因此项目状态已经从：

> Algorithm 1 / KeyMat 能接进系统，但内部层仍主要依赖显式 bridge

升级为：

> **KeyMat 已经实质性地吸收到核心模块参数与模块结构中，模型显著更接近论文意义上的 fully-obfuscated internal modules。**

---

## 13. 阶段 H：Attention 静态化收敛与噪声强度定标

## 13.1 阶段 H 目标

阶段 H 明确收窄为两件事：

1. **H2：噪声强度定标**
2. **H1：Attention 静态化收敛**

不做：

- 攻击实验
- 隐私参数扫描
- accuracy-privacy tradeoff 曲线

因此阶段 H 的本质是两类“收口工作”：

- **参数工作点收口**
- **部署形态收口**

## 13.2 H2：噪声强度定标

结果文件：

- `outputs/stage_h/noise_calibration.json`

本阶段覆盖：

- 论文默认点：`αe=1.0, αh=0.2, λ=0.3, h=128, β=8, γ=1e3`
- 一组粗扫 / 细定点：
  - `fine_a`
  - `fine_b`
  - `fine_c`
  - `mild_a`
  - `mild_b`
  - `mild_c`
  - `paper_lambda_low`
  - `paper_lambda_high`

### 论文默认点结果

- `avg_final_logits_restored_max_abs_error = 19.7789`
- `avg_layer_23_block_out_max_abs_error = 27.7285`
- `greedy_first_token_match_rate = 0.4`
- `generated_ids_exact_match_rate = 0.0`
- `generated_text_exact_match_rate = 0.0`

这说明：

> 在当前实现与当前模型规模下，论文默认噪声点明显过强，不能直接作为默认工作点。

### 当前推荐非零工作点

当前阶段 H 确定的推荐工作点为：

- `αe = 0.1`
- `αh = 0.05`
- `λ = 0.3`
- `h = 128`
- `β = 8`
- `γ = 1e3`

对应结果：

- `avg_final_logits_restored_max_abs_error = 2.6765`
- `avg_layer_23_block_out_max_abs_error = 1.7660`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 0.8`
- `generated_text_exact_match_rate = 0.8`

### H2 结论

阶段 H 现在已经有了两组明确可复用参数：

- **参考点**：`stable_reference`
- **推荐非零工作点**：`fine_a`

这意味着后续阶段不再需要反复猜噪声怎么设。

## 13.3 H1：Attention 静态化收敛

阶段 G 虽然已经完成 attention fused，但复杂 attention 内核仍保留了较强的运行时结构。

阶段 H 的目标是：

> 尽量把这些固定结构前移到离线权重构造中，让 runtime 主要只保留标准 attention 必需部分。

当前在 `src/stage_h_attention_static.py` 中，已经进一步前移并静态化了：

- q/k 的 intra-head 线性变换
- `τ_kv`
- `τ_group`
- `o_proj` 输入侧对应的 inverse permutation

而在线保留的部分仅包括：

- q/k/v 张量计算
- RoPE
- softmax
- attention 聚合
- 输出投影

### 关键排错

阶段 H 的静态 attention 并不是一次成功的，经历了两轮关键修正：

1. **bridge / static 比较 seed 不一致**
   - 早期 staticized candidate 使用了不同 seed，导致比较失真
2. **`o_weight` 吸收 query-head permutation 的方向写反**
   - 修正后 static attention 才真正回到 stage-G bridge baseline 同量级

### block0 结果

结果文件：

- `outputs/stage_h/attention_static_block0.json`

bridge baseline：

- `avg_final_logits_restored_max_abs_error = 3.4439`
- `avg_layer_0_attn_out_max_abs_error = 0.0177312`
- `avg_layer_0_block_out_max_abs_error = 0.1363455`

staticized candidate：

- `avg_final_logits_restored_max_abs_error = 3.4399`
- `avg_layer_0_attn_out_max_abs_error = 0.0177317`
- `avg_layer_0_block_out_max_abs_error = 0.1362447`

### prefix-2 / prefix-4 / full-layer 结果

结果文件：

- `outputs/stage_h/attention_static_prefix_2.json`
- `outputs/stage_h/attention_static_prefix_4.json`
- `outputs/stage_h/full_layers.json`

关键结果：

- prefix-2：
  - bridge `avg_layer_1_block_out_max_abs_error = 0.1312623`
  - static `= 0.1311570`
- prefix-4：
  - bridge `avg_layer_3_block_out_max_abs_error = 0.2525876`
  - static `= 0.2522361`
- full-layer：
  - bridge `avg_final_logits_restored_max_abs_error = 3.4455`
  - static `= 3.4421`
  - bridge `avg_layer_23_block_out_max_abs_error = 1.6600623`
  - static `= 1.6673660`

### H1 结论

阶段 H 已经证明：

> attention 的 runtime 复杂结构可以进一步收缩到离线权重表达中，而不显著劣化阶段 G 的恢复能力。

## 13.4 混淆模型交付与 server I/O 路径

阶段 H 已经导出一版更贴近当前最新实现的可交付工件：

- `artifacts/stage_h_full_obfuscated/metadata.json`
- `artifacts/stage_h_full_obfuscated/model_state.pt`
- `artifacts/stage_h_full_obfuscated/server_model_state.pt`
- `artifacts/stage_h_full_obfuscated/client_secret.pt`

当前大小约：

- `server_model_state.pt ≈ 4.1G`
- `client_secret.pt ≈ 2.4M`
- 目录总大小约 `8.1G`（含 demo bundle `model_state.pt`）

此外，当前还新增了一套更接近 `save_pretrained` 的导出布局：

- `artifacts/stage_h_pretrained/server/config.json`
- `artifacts/stage_h_pretrained/server/generation_config.json`
- `artifacts/stage_h_pretrained/server/tokenizer.json`
- `artifacts/stage_h_pretrained/server/tokenizer_config.json`
- `artifacts/stage_h_pretrained/server/chat_template.jinja`
- `artifacts/stage_h_pretrained/server/model.safetensors`
- `artifacts/stage_h_pretrained/server/obfuscation_config.json`
- `artifacts/stage_h_pretrained/client/client_secret.pt`

这套目录当前大小约：

- `4.3G`

这版工件已经实际验证可加载、可推理：

- 导出：`scripts/export_stage_h_model.py`
- 推理：`scripts/infer_stage_h_model.py`

### 能不能直接交付给 server？

**可以，但要明确当前交付形态。**

它不是 Hugging Face `save_pretrained` 风格的通用模型目录，而是：

> **依赖当前仓库重建逻辑的可部署工件**

也就是说，server 侧需要：

1. 当前仓库代码
2. 本地 baseline 模型目录
3. `artifacts/stage_h_full_obfuscated/server_model_state.pt`
4. `artifacts/stage_h_full_obfuscated/metadata.json`

然后通过：

- `src/stage_h_artifact.py`

里的加载逻辑重建并运行混淆模型。

客户端侧则应保留：

- `artifacts/stage_h_full_obfuscated/client_secret.pt`

这部分包含：

- `perm_vocab`
- `inv_perm_vocab`

因此更准确地说，当前已经有两种交付形态：

### 形态 A：演示 bundle

- `model_state.pt`

适合本地演示、完整重载、快速测试。

### 形态 B：client / server 分离

- server：
  - `server_model_state.pt`
  - `metadata.json`
- client：
  - `client_secret.pt`

这一形态更接近论文里的“server 只持有混淆模型，client 持有输入/输出映射秘密”的部署方式。

### 形态 C：pretrained-like bundle

- server：
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `chat_template.jinja`
  - `model.safetensors`
  - `obfuscation_config.json`
- client：
  - `client_secret.pt`

对应脚本：

- 导出：`scripts/export_stage_h_pretrained.py`
- 推理：`scripts/infer_stage_h_pretrained.py`

这一形态仍需要当前仓库里的自定义 loader，但目录布局已经明显更接近 `save_pretrained`。

### 输入“加密”在哪里做？

当前输入侧的 token-level 混淆在：

- `src/transforms.py`
  - `map_input_ids(...)`

运行时调用位置体现在各阶段推理脚本和工件推理脚本里，例如：

- `scripts/infer_stage_h_model.py`

逻辑是：

1. 客户端先把原始 `input_ids`
2. 用 `perm_vocab`
3. 映射成混淆 token ids
4. 再把这些混淆 token ids 发送给 server 模型

### 输出“解密”在哪里做？

当前输出侧的恢复在：

- `src/transforms.py`
  - `restore_logits(...)`
  - `unmap_output_ids(...)`

当前主推理链路里主要使用的是：

- 服务器输出混淆词表空间 logits
- 客户端用 `restore_logits(...)` 恢复到原词表空间
- 再做 argmax / decode

这也是 `scripts/infer_stage_h_model.py` 当前采用的方式。

### 当前这版 server 工件的最准确定义

> **已经有一版“可以直接交付给 server，在当前仓库内重建并运行”的混淆模型工件；输入映射与输出恢复仍在客户端侧完成。**

## 13.5 阶段 H 结论

阶段 H 当前可以正式判定为：

1. 噪声强度定标完成
2. 论文默认参数点已实际验证
3. 推荐非零工作点已确定
4. attention 静态化收敛完成
5. stage-H 混淆模型工件已导出并验证可推理

因此当前项目状态可以更准确地描述为：

> **已经完成 AloePri 技术报告第 5 节方法实现的复现，并进一步把 attention 的工程表达向论文离线混淆部署形态收敛；同时完成了 embed/head 噪声强度的功能定标，但尚未进入第 6 节的隐私攻击评估。**

---

## 14. 阶段 I：标准 HF/vLLM 入口与 Phase 2 阻塞定位

阶段 I 的角色已经从旧口径中的“泛下一步”收紧为：

> **先把标准 HF/vLLM 入口打通，并明确继续物化到标准 checkpoint 时到底卡在哪里。**

本阶段已经完成：

- Stage A 标准 HF checkpoint 导出
- client/server 映射契约
- HF 路径 FP32 / BF16 严格回归
- vLLM 回归脚本实现
- Phase 2 的 shape/materialization probe

关键结果：

- `artifacts/stage_i_vllm/server/` 可被 `AutoModelForCausalLM.from_pretrained(...)` 直接加载
- `outputs/stage_i/hf_regression_fp32.json` 与 `outputs/stage_i/hf_regression_bf16.json` 都达到：
  - `avg_full_logits_max_abs_error = 0.0`
  - `generated_ids/text exact match = 1.0`
- `outputs/stage_i/phase2_probe.json` 证明当前扩维 KeyMat 与标准 checkpoint 形状存在结构性冲突：
  - `896 -> 1152`
  - `embedding/head/qkv/ffn/norm` shape 全部不再匹配原始 Qwen2

同时，阶段 I 也把 vLLM 阻塞进一步收缩成：

- 当前本机不是 checkpoint 形状有问题
- 而是 **vLLM CPU backend 环境没有真正跑起来**

结论：

> **阶段 I 已完成“标准入口 + 问题定位”这两件事。**

---

## 15. 阶段 J：standard-shape full-layer 协变恢复

阶段 J 承接阶段 I 的阻塞点，转向：

> **放弃扩维 KeyMat 作为标准 checkpoint 目标，改用 non-expanding / standard-shape hidden transform，并把它从 `embed/head-only` 推进到 full-layer。**

本阶段已经完成：

- `prefix-1`
- `prefix-2`
- `prefix-4`
- `prefix-8`
- `full-layer`

并且不仅在内存里成立，还导出了一版 full-layer 标准 checkpoint：

- `artifacts/stage_j_full_square/`

核心结果：

- `outputs/stage_j/prefix_1_square.json`
  - `avg_final_logits_restored_max_abs_error ≈ 1.20e-4`
- `outputs/stage_j/prefix_2_square.json`
  - `avg_final_logits_restored_max_abs_error ≈ 1.75e-4`
- `outputs/stage_j/prefix_4_square.json`
  - `avg_final_logits_restored_max_abs_error ≈ 1.34e-4`
- `outputs/stage_j/prefix_8_square.json`
  - `avg_final_logits_restored_max_abs_error ≈ 1.91e-4`
- `outputs/stage_j/full_layers_square.json`
  - `avg_final_logits_restored_max_abs_error ≈ 1.26e-4`
  - `generated_ids/text exact match = 1.0`
- `outputs/stage_j/full_layers_square_export_regression.json`
  - 导出后仍保持：
    - `avg_full_logits_max_abs_error ≈ 1.26e-4`
    - `generated_ids/text exact match = 1.0`

这一步的意义非常大，因为它证明：

> **standard-shape 路线不只是“shape 对了”，而是真的把 full-layer 功能恢复做通了。**

---

## 16. 阶段 J 补充：standard-shape full-layer 噪声定标

在 standard-shape full-layer 打通后，又进一步做了非零噪声定标。

结果文件：

- `outputs/stage_j/noise_calibration.json`

当前推荐的非零工作点是：

- `alpha_e = 0.02`
- `alpha_h = 0.01`

对应 profile 名称：

- `tiny_a`

关键结果：

- `avg_final_logits_restored_max_abs_error ≈ 0.672`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

并且已经导出对应工件：

- `artifacts/stage_j_full_square_tiny_a/`

以及导出后回归：

- `outputs/stage_j/full_layers_square_tiny_a_export_regression.json`

说明：

> **standard-shape 路线不仅 zero-noise 可用，而且已经找到了一组稳定的非零噪声工作点。**

---

## 17. 阶段 K：standard-shape 工件统一交付包装

在 `stage_j_full_square` 与 `stage_j_full_square_tiny_a` 都稳定之后，项目已经从“模型正确”推进到“工件可交付”。

阶段 K 当前已完成：

- 统一发布目录：
  - `artifacts/stage_k_release/`
- profile catalog：
  - `stable_reference`
  - `tiny_a`
- 统一推理入口：
  - `scripts/infer_stage_k_release.py`

当前目录结构：

```text
artifacts/stage_k_release/
├── catalog.json
├── deployment_contract.json
├── README.md
└── profiles/
    ├── stable_reference -> artifacts/stage_j_full_square
    └── tiny_a -> artifacts/stage_j_full_square_tiny_a
```

实测：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py --profile tiny_a --prompt "请用一句话介绍你自己。" --max-new-tokens 8
```

输出正常：

- `我是一个人工智能助手，可以帮助您解答`

结论：

> **阶段 K 已经把 standard-shape full-layer 路线整理成“有 profile、可选配置、可直接推理”的交付形态。**

补充说明：

- 当前阶段 K 的这一轮完成，直接对应的是 **Qwen standard-shape 路线**
- 也就是：
  - `stage_j_full_square`
  - `stage_j_full_square_tiny_a`
- `Llama-3.2-3B` 当前已经完成 Stage I / Stage J 和真实 4090 correctness 验证，但**尚未独立完成 Llama 专属的 Stage K release 包装**

---

## 18. Qwen 与 Llama 路线区分

当前仓库中同时存在两条相关但不完全相同的路线：

### Qwen 主线

Qwen 是当前仓库最完整的主线，已经完成：

- 阶段 A～H：主方法链
- 阶段 I：标准入口
- 阶段 J：standard-shape full-layer
- 阶段 J（噪声）：非零工作点定标
- 阶段 K：统一交付包装

因此当前 Qwen 路线已经达到：

> **研究复现主链 + standard-shape full-layer + release 交付包装**

### Llama-3.2-3B 扩展线

Llama 当前不是从 A 开始重新独立推进，而是在 Qwen 主线已成立后，优先完成：

- `LlamaArchitectureAdapter`
- Stage I 标准 HF 工件
- Stage J standard-shape full-layer
- 云端真实 4090 correctness 验证

当前 Llama 已完成：

- 本机 mock 闭环
- 云端真实 Stage I artifact sanity / remote validation
- 云端真实 Stage J remote validation
- 云端真实 Stage J 噪声定标
- Llama 专属 Stage K release 导出

当前 Llama 尚未独立完成：

- `vLLM` 实跑
- 更系统的安全评估

但当前已经具备继续推进所需的脚本：

- `scripts/run_stage_j_llama_real_noise_calibration.py`
- `scripts/export_stage_k_llama_release.py`
- `scripts/run_llama_3b_stagek_pipeline.sh`

更准确的项目状态表述应为：

> **Qwen 主线已经完成 A–K；Llama-3.2-3B 也已经完成结构接入、Stage I、Stage J、真实 4090 correctness 验证、真实噪声定标与独立 Stage K release。**

相关补充文档：

- `docs/Qwen与Llama复现阶段区分说明.md`
- `docs/Llama-3.2-3B标准形状恢复报告.md`

---

## 19. 当前复现与论文完整版的差异

尽管已经做到阶段 H，但当前实现仍然不是论文完整版 AloePri。

### 当前已经实现的

- 词表置换闭环
- hidden-space 简化 key（permutation + scaling）
- attention 基础协变恢复
- RMSNorm `κ` 补偿
- FFN 中间维 permutation + scaling
- block0 完整恢复
- 多层 block 级恢复验证
- 论文第 5.2.3 节复杂 attention 结构补全与修复
- 论文第 5.2.1 的 Algorithm 1 / KeyMat 原生矩阵生成
- 论文第 5.2.2 的 embed/head KeyMat 路径与噪声实验
- KeyMat 下的 norm / attention / FFN 显式 bridge
- KeyMat 的 block0 / prefix / full-layer 功能验证
- KeyMat 的 fused RMSNorm / FFN / Attention
- KeyMat 融合化后的 prefix-2 / prefix-4 / full-layer 回归
- 一版可直接演示推理的 stage-G 混淆模型导出工件
- attention 的进一步静态化收敛
- embed/head 噪声强度的功能定标
- 一版可直接交付 server 的 stage-H 混淆模型工件

### 当前尚未实现的

- Hugging Face `save_pretrained` 风格的通用混淆模型导出格式
- 更进一步压缩 runtime 结构、减少重建依赖的部署形态
- 论文更完整的动态 `Ẑ_block` 采样细节与更广泛消融
- 更贴近论文攻击面和安全评估的实验

因此，当前工作的准确定位应是：

> **从简化 AloePri 路线推进到 Algorithm 1 / KeyMat 原生体系，并完成内部层参数级融合、attention 静态化收敛与系统验证**

而不是：

> **完整工业版 AloePri 复现**

---

## 20. 阶段 A～K 的总体结论

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

### 阶段 F

达成：

- **Algorithm 1 / KeyMat 原生体系首次系统接入**

结论：

- `P @ Q ≈ I` 数值验收通过；
- embed/head、norm、attention、FFN 已经在 KeyMat bridge 下打通；
- block0、prefix-2、prefix-4、full-layer 都保持近乎精确恢复；
- 说明论文原生 KeyMat 已能稳定接入当前 A–E 主线。

### 阶段 G

达成：

- **KeyMat 融合化 / 去 bridge 化**

结论：

- fused RMSNorm、FFN、Attention 已全部打通；
- prefix-2 / prefix-4 / full-layer 回归全部通过；
- full-layer fused 结果与阶段 F bridge 基线保持同级别；
- 已导出一版可直接演示推理的 stage-G 混淆模型工件。

### 阶段 H

达成：

- **Attention 静态化收敛与噪声强度定标**

结论：

- 论文默认噪声点已验证，当前推荐非零工作点已确定；
- attention 中更多固定结构已前移到离线权重表达；
- staticized attention 在 block0 / prefix-2 / prefix-4 / full-layer 上均保持阶段 G 同量级恢复能力；
- 已导出一版可直接交付 server 并在当前仓库内重建运行的 stage-H 混淆模型工件。

### 阶段 I

达成：

- **标准 HF/vLLM 入口打通与阻塞定位**

结论：

- Stage A 标准 checkpoint 导出与 HF 回归严格成立；
- Phase 2 的 shape 阻塞已定位到扩维 KeyMat；
- 当前 vLLM 真正未打通的原因已收敛为 CPU backend 环境问题，而非 checkpoint 本身。

### 阶段 J

达成：

- **standard-shape full-layer 协变恢复**

结论：

- non-expanding hidden transform 已从 `embed/head-only` 推进到 `prefix-1 / 2 / 4 / 8 / full-layer`；
- full-layer 标准 checkpoint 已导出并重新回归通过；
- standard-shape 路线已经被证明在功能上成立。

### 阶段 J（噪声）

达成：

- **standard-shape full-layer 噪声定标**

结论：

- 推荐非零工作点已确定为 `tiny_a = (alpha_e=0.02, alpha_h=0.01)`；
- 该配置在导出前后都保持 `generated_ids/text exact match = 1.0`。

### 阶段 K

达成：

- **standard-shape 工件统一交付包装**

结论：

- `stable_reference` 与 `tiny_a` 已收拢到统一发布目录；
- 已有统一 catalog / deployment contract / inference CLI；
- 项目已经从“模型可用”推进到“可交付使用”。

---

## 21. 当前正式判定

基于当前全部实验和数据，可以给出如下正式判定：

> **当前仓库已经完成 AloePri 技术报告的一条从“简化主线”推进到“standard-shape full-layer 可交付形态”的复现路线：从词表闭环，到 attention 子层恢复，到 block 完整恢复，再到多层传播验证、复杂 attention 结构接入修复、Algorithm 1 / KeyMat 原生体系接入、内部层 KeyMat 融合化、attention 静态化收敛与噪声定标，并进一步完成了 standard HF 入口、non-expanding full-layer 恢复、full-layer 非零噪声工作点定标，以及统一交付包装。**

更精确地说：

- 阶段 A：严格完成
- 阶段 B：严格完成
- 阶段 C：严格完成
- 阶段 D：完成多层传播验证，并首次观察到系统级回归
- 阶段 E：完成复杂 attention 结构接入、核心 bug 修复与功能回归
- 阶段 F：完成 Algorithm 1 / KeyMat 原生体系的首次系统功能接入
- 阶段 G：完成 KeyMat 的参数级融合化与系统级去 bridge 回归
- 阶段 H：完成 attention 静态化收敛、噪声工作点定标与 stage-H 工件交付
- 阶段 I：完成标准 HF/vLLM 入口打通与 Phase 2 阻塞定位
- 阶段 J：完成 standard-shape full-layer 恢复与标准 checkpoint 导出
- 阶段 J（噪声）：完成 standard-shape full-layer 噪声定标
- 阶段 K：完成 standard-shape 工件统一交付包装

---

## 22. 下一步建议

当前最合理的下一步已经不再是继续验证“standard-shape 路线是否可行”，因为这件事已经完成。现在更合理的方向是：

- 继续攻克 vLLM CPU backend 环境，验证 `stage_j_full_square` / `tiny_a` 在 vLLM 侧的真正实跑；
- 或进入论文第 6 节口径的安全性与攻击评估；
- 或继续把阶段 K 做成更完整的 materialized copy / baseline-free loader / 服务化接口。

建议顺序：

### 阶段 I

收紧为“**标准 HF/vLLM 入口打通**”：

- Stage A 标准 checkpoint 导出
- client/server 输入输出映射契约
- HF correctness 回归
- vLLM 可运行性验证
- 并识别继续走标准 shape 物化时的真实阻塞点

### 阶段 J

承接阶段 I 的阻塞点，做“**标准形状协变恢复**”：

- 使用 non-expanding hidden transform
- 从 embed/head-only 推进到 block0
- 再推进到 prefix-2 / prefix-4 / prefix-8 / full-layer
- 当前已产出一版 `artifacts/stage_j_full_square/` 标准 checkpoint
- 目标是在保持标准 checkpoint shape 的前提下恢复功能正确性

### 阶段 K

在 I + J 都成立后，再做“**更完整的交付与服务化包装**”：

- 更通用的混淆模型导出格式
- baseline-free loader
- 更接近原生 `save_pretrained` / 服务接口的分发与部署体验

当前补充判断：

- 阶段 J 这条 standard-shape 路线已经在 HF 路径下跑通到 full-layer，并完成非零噪声定标；
- 阶段 K 也已经完成第一轮交付包装；
- 因此当前真正剩余的阻塞点，已经进一步收敛为：
  - **vLLM CPU backend 环境**
  - **安全评估 / 攻击实验尚未启动**
  - **更深的产品化包装尚未启动**

---

## 23. 结语

当前复现路线的最大价值不在于“已经完整复现论文全部工程细节”，而在于：

- 它已经把论文里最难调试的一条主线——**协变恢复链路**——从最小闭环一直推进到了多层传播；
- 每一步都有实验与数据支撑，不是只靠理论推断；
- 并且已经明确区分了：
  - 当前是简化版什么；
  - 还缺论文完整版什么；
  - 下一步最值得补什么。

因此，这份复现工作已经具备很强的研究与工程价值：  
它不仅验证了论文思路在小模型上的可行性，还已经把复杂 attention 与原生 KeyMat 体系都接进了统一实验链路，为后续的 KeyMat 融合化与攻击评估提供了一个稳定、可解释、可迭代的基线。
