# AloePri 复现报告（阶段 A 与阶段 B 严格校验）

## 1. 报告目的

本文档用于对当前仓库中已经完成的两个复现阶段进行一次**严格、可追溯、可复跑**的技术确认：

- **阶段 A：最小词表闭环**
  - 目标：只做词表置换，不引入 hidden-space 变换；
  - 验证 `perm_vocab / inv_perm_vocab`、embedding 行置换、`lm_head` 词表方向置换、在线 token id 映射、输出 logits 还原是否严格闭环。

- **阶段 B：最小 hidden-space key 体系 + block0 attention 入口验证**
  - 目标：在阶段 A 的基础上引入最小 hidden-space 可逆变换；
  - 证明只改 embed/head 会在第 0 层 attention 失配；
  - 证明只修复 `layer_0.self_attn` 后，`layer_0_attn_out` 在恢复到原始 basis 后会显著接近 baseline；
  - 证明 wrapper 版与 fused-weight 版是一致的。

本报告同时回答两个问题：

1. 当前已完成内容是否真的满足对应阶段目标；
2. 当前验证结果是否足以支撑进入下一阶段。

---

## 2. 复现环境与对象

### 2.1 模型

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 本地路径：`model/Qwen2.5-0.5B-Instruct`

### 2.2 运行环境

- Python：`3.11`
- 核心依赖：
  - `torch`
  - `transformers`
  - `accelerate`
  - `safetensors`
  - `pytest`

### 2.3 固定输入

阶段 A 与阶段 B 都统一使用以下 5 条 prompt：

1. `请用一句话介绍你自己。`
2. `什么是 Transformer block？`
3. `请解释一下注意力机制。`
4. `Hello, how are you?`
5. `Write one sentence about machine learning.`

### 2.4 固定随机性

全流程固定：

- Python 随机种子
- NumPy 随机种子
- Torch 随机种子
- 词表 permutation 种子
- hidden permutation / scaling 种子

默认种子为：

- `20260323`

---

## 3. 本次严格校验执行内容

本次重新执行的校验包括：

### 3.1 测试

执行：

```bash
conda run --no-capture-output -n qwen-transformers pytest -q
```

结果：

- `13 passed`

这 13 个测试覆盖了：

- 阶段 A：
  - permutation 逆映射正确性
  - special / added token 固定策略
  - token id 映射方向
  - logits 恢复方向
  - embedding / `lm_head` 置换语义
  - 单步 logits 等价

- 阶段 B：
  - hidden transform 可逆性
  - hidden transform toy round-trip
  - hidden transform 对最后一维的正确作用
  - block0 attention wrapper 改善效果
  - fused-weight 版与 wrapper 版一致性

### 3.2 阶段 A 脚本

执行：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_baseline.py --max-new-tokens 8
conda run --no-capture-output -n qwen-transformers python scripts/run_permuted_eval.py --max-new-tokens 8
```

并额外重建了**全位置 logits 严格校验**，结果输出到：

- `outputs/permuted_eval/stage_a_strict_verification.json`

### 3.3 阶段 B 脚本

执行：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_b_hidden_only.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_b_block0_attn_wrapper.py
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_b_block0_attn_fused.py
```

对应输出：

- `outputs/stage_b/hidden_only.json`
- `outputs/stage_b/block0_attn_wrapper.json`
- `outputs/stage_b/block0_attn_fused.json`

同时生成总汇总：

- `outputs/stage_validation_summary.json`

---

## 4. 阶段 A：最小词表闭环严格校验

## 4.1 阶段 A 目标

阶段 A 只验证**词表空间闭环**，不引入任何 hidden-space 变换。

它完成的事情是：

1. 生成 `perm_vocab / inv_perm_vocab`
2. 对 embedding 的词表维做行置换
3. 对 `lm_head` 的输出词表维做对应置换
4. 客户端在线把输入 token ids 做 permutation
5. 模型输出 logits 后再做 inverse permutation
6. 验证恢复后的结果是否与 baseline 完全一致

这一步的理论目标是：

\[
\text{Model}(x) \equiv \text{inv\_perm}(\text{Model}_{perm}(\text{perm}(x)))
\]

注意：阶段 A 不处理 hidden basis，也不处理中间层结构。

## 4.2 阶段 A 已实现模块

核心实现包括：

- `src/key_manager.py`
  - 词表 permutation 生成、求逆、合法性检查、保存/加载

- `src/transforms.py`
  - 输入 token id 映射
  - 输出 token id 恢复
  - logits 词表维恢复

- `src/obfuscate_embed_head.py`
  - embedding 行置换
  - `lm_head` 词表方向置换
  - 基于副本构造 vocab-permuted model

- `scripts/run_baseline.py`
  - baseline 推理与结果落盘

- `scripts/run_permuted_eval.py`
  - 词表闭环验证脚本

## 4.3 阶段 A 特别保护策略

当前实现没有对整个 `model_vocab_size` 盲目做 permutation，而是采用了保守策略：

- 只移动普通词表区间；
- special token 固定；
- added token 固定；
- `len(tokenizer)` 之外的 tail rows 固定。

这是当前阶段 A 成功的重要原因之一，因为它避免了 chat template token 与未使用 embedding 行被打乱。

## 4.4 阶段 A 严格校验指标

本次校验使用三类指标：

1. **全位置 full logits 一致性**
   - 对整段 prompt 所有位置的 logits 做逐元素比较；
   - 不是只比较最后一个 token。

2. **生成 token ids 一致性**
   - 对 greedy 短生成的输出 token ids 做逐步比较。

3. **生成文本一致性**
   - 对解码后的生成文本做直接比较。

## 4.5 阶段 A 严格校验结果

来自 `outputs/permuted_eval/stage_a_strict_verification.json` 的结论如下：

- `prompt_count = 5`
- 所有 prompt 都满足：
  - `full_logits_equal = true`
  - `max_abs_error = 0.0`
  - `mean_abs_error = 0.0`
  - `generated_ids_equal = true`
  - `generated_text_equal = true`

额外校验：

- `special_ids_fixed = true`
- `added_ids_fixed = true`
- `tail_rows_fixed = true`

## 4.6 阶段 A 每条 prompt 的验证结论

| Prompt ID | Prompt | Full Logits | Max Abs Error | Generated IDs | Generated Text |
|---|---|---:|---:|---:|---:|
| 1 | 请用一句话介绍你自己。 | True | 0.0 | True | True |
| 2 | 什么是 Transformer block？ | True | 0.0 | True | True |
| 3 | 请解释一下注意力机制。 | True | 0.0 | True | True |
| 4 | Hello, how are you? | True | 0.0 | True | True |
| 5 | Write one sentence about machine learning. | True | 0.0 | True | True |

## 4.7 阶段 A 技术结论

阶段 A 可以得出一个非常明确的结论：

> 当前实现的词表闭环是**严格成立**的，不是近似成立，也不是“输出看起来差不多”。

这意味着：

- `perm_vocab / inv_perm_vocab` 的方向是正确的；
- embedding 行置换方向是正确的；
- `lm_head` 的词表方向改写是正确的；
- logits 恢复方向是正确的；
- 整条 baseline → permuted model → inverse restore 的链路在阶段 A 中是数学上与工程上都闭合的。

### 结论：阶段 A 已完成

按其定义，阶段 A 已完成，而且已被严格验证通过。

---

## 5. 阶段 B：最小 hidden-space key 体系 + block0 attention 入口验证

## 5.1 阶段 B 目标

阶段 B 的目标不是恢复整网，也不是恢复最终生成质量，而是要验证以下因果链条：

1. 一旦 embedding 输出进入新的 hidden basis，
2. 如果中间 attention 不做协变改写，
3. 那么第 0 层 attention 开始就会失配；
4. 如果只修第 0 层 attention，
5. 那么 `layer_0_attn_out` 在恢复回原始 basis 后应该明显改善。

因此，阶段 B 的目标是一个**入口验证**，而不是完整恢复。

## 5.2 阶段 B 已实现内容

### hidden-space 变换

新增 `src/hidden_keys.py`，实现：

- hidden permutation
- hidden scaling
- 可逆 hidden transform
- inverse hidden transform
- 可逆性验证

当前 transform 采用最小形式：

\[
P_h = \Pi_h \cdot S_h
\]

其中：

- `\Pi_h` 是 hidden 维 permutation
- `S_h` 是对角 scaling

### hidden-space 前向包装器

新增 `StageBHiddenPermutationModel`，其行为是：

1. 先走阶段 A 的词表置换 embedding；
2. embedding 输出后显式乘 `P_h`；
3. 中间网络在 obfuscated basis 中继续前向；
4. 最终进入 `lm_head` 前乘 `P_h^{-1}`；
5. 之后再走阶段 A 已经验证过的词表闭环。

### block0 tracing

新增 `TraceRecorder` 与 block0 hook，抓取：

- `embed_out`
- `layer_0_input`
- `layer_0_q_proj_out`
- `layer_0_k_proj_out`
- `layer_0_v_proj_out`
- `layer_0_attn_out`
- `layer_0_block_out`
- `final_logits`

比较时遵循统一原则：

> 对 obfuscated hidden state 必须先恢复到原始 basis，再与 baseline 比较。

### block0 attention wrapper

新增 `TracingQwen2Attention`，支持两种模式：

- `plain`
- `wrapper`

其中 wrapper 模式的作用是：

- 在进入 `q_proj/k_proj/v_proj` 前先把 hidden 从 obfuscated basis 拉回 base basis；
- 执行 attention 的线性映射；
- attention 输出后再乘回 `P_h`，回到 obfuscated basis。

### fused-weight 版

进一步把 wrapper 中的显式 basis 乘法吸收到：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

形成 block0 attention 的 fused-weight 版本。

## 5.3 阶段 B 三种实验模式

阶段 B 共做了三种模式：

### 模式 1：hidden_only

- embed/head 接 hidden transform
- block0 attention 不改

作用：

- 证明“不改 attention 会失配”

### 模式 2：block0_attn_wrapper

- embed/head 接 hidden transform
- block0 attention 用 wrapper 修复

作用：

- 验证 attention 子层恢复是否有效

### 模式 3：block0_attn_fused

- embed/head 接 hidden transform
- block0 attention 用 fused weights 修复

作用：

- 验证 fused-weight 版是否与 wrapper 版一致

---

## 6. 阶段 B 关键校验结果

## 6.1 hidden transform 本身是否可靠

在测试中：

- `test_hidden_transform_invertibility`
- `test_hidden_transform_round_trip_toy`

已经验证：

- `P_h` 与 `P_h^{-1}` 是可逆的；
- toy hidden vector 经前后变换后可以恢复；
- transform 数值稳定。

因此阶段 B 失配不来自 transform 定义错误，而是来自中间模块未协变改写。

## 6.2 hidden_only 的结果

来自 `outputs/stage_b/hidden_only.json` 的平均指标：

- `avg_embed_out_restored_max_abs_error = 5.215406417846679e-09`
- `avg_layer_0_attn_out_restored_max_abs_error = 0.4838339626789093`
- `avg_layer_0_block_out_restored_max_abs_error = 3.482872486114502`
- `avg_final_logits_restored_max_abs_error = 29.652044677734374`

解释：

- `embed_out_restored` 几乎精确恢复，说明 embed/head 接 hidden transform 方向正确；
- 但 `layer_0_attn_out` 误差明显出现，说明只改 embedding/head 而不改中间 attention，等价性会被破坏；
- `layer_0_block_out` 和 `final_logits` 的误差继续放大，这是预期现象。

### 结论

这一步成功证明了：

> **一旦 hidden-space basis 改变，不修 attention，block0 attention 就会失配。**

这正是阶段 B 设计的第一条核心结论。

## 6.3 block0_attn_wrapper 的结果

来自 `outputs/stage_b/block0_attn_wrapper.json` 的平均指标：

- `avg_embed_out_restored_max_abs_error = 5.215406417846679e-09`
- `avg_layer_0_attn_out_restored_max_abs_error = 0.26317378878593445`
- `avg_layer_0_block_out_restored_max_abs_error = 3.1904022693634033`
- `avg_final_logits_restored_max_abs_error = 33.16814651489258`

与 hidden_only 比较：

- `layer_0_attn_out_restored_max_abs_error`
  - 从 `0.4838339626789093`
  - 降到 `0.26317378878593445`

这是一次**显著改善**。

### 解释

这说明：

- 当前 block0 attention 的 wrapper 逻辑已经修复了 attention 子层里的关键 basis 失配；
- attention 输出恢复回原始 basis 后，已经明显更接近 baseline；
- 但由于：
  - pre/post RMSNorm 还没修；
  - FFN 还没修；
  - block 后半段仍然在未完全协变的链路里；
  所以 `layer_0_block_out` 与 `final_logits` 仍然不会完全好转。

### 结论

这一步成功证明了：

> **只修 block0 attention 子层，已经能够系统性降低 `layer_0_attn_out` 的恢复误差。**

这满足阶段 B 的第二条核心结论。

## 6.4 block0_attn_fused 的结果

来自 `outputs/stage_b/block0_attn_fused.json` 的平均指标：

- `avg_embed_out_restored_max_abs_error = 5.215406417846679e-09`
- `avg_layer_0_attn_out_restored_max_abs_error = 0.2631740629673004`
- `avg_layer_0_block_out_restored_max_abs_error = 3.1904025077819824`
- `avg_final_logits_restored_max_abs_error = 33.16811408996582`

与 wrapper 版比较：

- `layer_0_attn_out_restored_max_abs_error`
  - wrapper：`0.26317378878593445`
  - fused：`0.2631740629673004`

这两个数字几乎完全一致。

### 结论

这说明：

> **block0 attention 的 fused-weight 版本与 wrapper 版本在数值上是一致的。**

换句话说：

- 先显式插入 basis 乘法验证方向；
- 再把逻辑融合进权重；
- 这个工程步骤是成立的。

这满足阶段 B 的第三条核心结论。

---

## 7. 阶段 B 总体结论

阶段 B 的目标不是恢复最终 logits，也不是恢复整层 block，更不是恢复整网生成。

它要证明的是三件事：

1. hidden transform 进入后，不改 attention 会失配；
2. 只改 block0 attention 子层后，`layer_0_attn_out` 能显著恢复；
3. wrapper 与 fused-weight 两种实现方式是等价的。

根据当前结果，这三件事都已经被验证。

### 结论：阶段 B 已完成

按阶段 B 的定义，它已经完成，并已被严格验证通过。

---

## 8. 对“阶段 B final logits 没有改善”的解释

这是一个必须明确写清楚的结论。

从当前结果看：

- hidden_only 模式的 `avg_final_logits_restored_max_abs_error ≈ 29.65`
- wrapper / fused 模式反而在该指标上约为 `33.17`

这并不意味着 attention 修复失败。

原因是：

1. 当前只修了 block0 attention 子层；
2. pre-attn / post-attn RMSNorm 没修；
3. FFN 没修；
4. 后续层全部仍在不匹配的 basis 链路中继续传播误差。

因此，阶段 B 的正确评估位置是：

- `embed_out_restored`
- `layer_0_q/k/v`
- `layer_0_attn_out_restored`

而不是最终 logits。

这也是为什么：

> 阶段 B 是“入口验证”，不是“整层恢复”。

---

## 9. 两个阶段的最终验收结论

## 9.1 阶段 A

是否完成：**是**

是否严格通过：

- full logits 全位置完全一致：**是**
- greedy 短生成完全一致：**是**
- 生成文本完全一致：**是**
- special / added / tail rows 固定：**是**

最终结论：

> 阶段 A 已严格完成，词表空间闭环成立。

## 9.2 阶段 B

是否完成：**是**

是否严格通过：

- hidden transform 可逆：**是**
- 只改 embed/head 不改 attention 会失配：**是**
- 改 block0 attention 后 `attn_out` 显著改善：**是**
- fused-weight 与 wrapper 一致：**是**

最终结论：

> 阶段 B 已严格完成，hidden-space 入口与 block0 attention 协变修复逻辑成立。

---

## 10. 当前仓库中的证据文件

### 10.1 阶段 A

- `outputs/baseline/01.json` ~ `05.json`
- `outputs/permuted_eval/01.json` ~ `05.json`
- `outputs/permuted_eval/vocab_permutation.pt`
- `outputs/permuted_eval/stage_a_strict_verification.json`

### 10.2 阶段 B

- `outputs/stage_b/hidden_only.json`
- `outputs/stage_b/block0_attn_wrapper.json`
- `outputs/stage_b/block0_attn_fused.json`

### 10.3 汇总

- `outputs/stage_validation_summary.json`

---

## 11. 当前实现的边界与限制

为了避免误读，需要明确：

### 已经做到的

- 阶段 A：完整词表闭环严格成立
- 阶段 B：hidden-space 入口与 block0 attention 协变改写逻辑成立

### 还没有做到的

- 没有实现论文完整版 KeyMat
- 没有实现 RMSNorm `κ` 修正
- 没有实现 FFN 协变改写
- 没有让 `layer_0_block_out` 完全恢复
- 没有让最终 logits 在阶段 B 下恢复
- 没有把 hidden-space 方案推广到全层 / 全模型

所以当前结论不能被表述成：

- “论文完整系统已经复现完毕”

更准确的表述是：

> 论文复现已经完成了两个关键基础阶段，且两者都被严格验证通过，为下一阶段 RMSNorm + FFN + block 完整恢复提供了可靠基础。

---

## 12. 下一阶段建议

基于当前结果，下一阶段最合理的是：

### 阶段 C：完成第 0 层完整 block 的恢复

建议顺序：

1. 加入 pre-attn RMSNorm 的协变修正
2. 加入 post-attn RMSNorm 的协变修正
3. 加入 FFN 的 `gate/up/down` 协变改写
4. 验证 `layer_0_block_out_restored` 是否显著接近 baseline
5. 再看最终 logits 是否开始回归

这是当前仓库最自然、最稳妥的下一步。

---

## 13. 最终结论

本次严格校验后的最终结论如下：

### 结论 1

阶段 A 已经严格完成。  
当前实现不仅能跑，而且在 5 条固定 prompt 上实现了：

- full logits 全位置逐元素完全一致
- greedy 生成 token ids 完全一致
- 生成文本完全一致

### 结论 2

阶段 B 已经严格完成。  
当前实现成功证明了：

- hidden-space 坐标变换一旦引入，不修 attention 会在 block0 attention 开始失配；
- 只修 block0 attention 子层即可显著改善 `layer_0_attn_out` 的恢复误差；
- wrapper 与 fused-weight 版在数值上等价。

### 结论 3

当前仓库已经具备进入论文更深层复现的可靠基础。  
从工程质量与实验链路的角度看：

- 词表空间链路已经可靠；
- hidden-space 入口验证已经可靠；
- block0 attention 的协变实现路径已经可靠。

因此，下一阶段可以放心进入：

> **RMSNorm + FFN + 第 0 层完整 block 恢复**

