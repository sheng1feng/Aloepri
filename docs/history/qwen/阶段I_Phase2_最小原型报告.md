# 阶段 I / Phase 2 最小原型报告

本文档记录 Phase 2 的第一个最小原型：

1. 实现 **square monomial hidden transform**
2. 只打通 **embed/head-only**
3. 导出一版新的标准 HF checkpoint
4. 跑 HF 回归验证 shape 和功能

这一步的目标不是“立刻做成可用系统”，而是验证下面两个最关键的问题：

- **标准 checkpoint shape 能不能保持？**
- **如果只改 embed/head，而不继续适配中间层，功能会变成什么样？**

---

## 1. 本轮新增实现

### 1.1 新增 square monomial transform

文件：

- `src/stage_i_square.py`

核心对象：

- `SquareMonomialTransform`

定义：

\[
P = c \cdot \Pi \cdot S_{\pm}
\]

其中：

- `Π`：hidden 维 permutation
- `S±`：逐维 `±1` sign
- `c`：全局常数（本轮默认 `1.0`）

对应也实现了：

- `P`
- `Q = P^{-1}`
- `P @ Q = I` 的数值检查

### 1.2 新增 embed/head-only 导出原型

文件：

- `scripts/export_stage_i_phase2_square_checkpoint.py`

做法：

1. 先构造 Stage A 模型（词表 permutation）
2. 再对 Stage A 的 embedding/head 做 square transform：
   - `W_embed' = W_embed_stageA @ P`
   - `W_head' = W_head_stageA @ Q^T`
3. 直接导出标准 HF checkpoint 到：

```text
artifacts/stage_i_phase2_square/
├── server/
└── client/
```

### 1.3 新增 HF 回归脚本

文件：

- `scripts/run_stage_i_phase2_square_regression.py`

它会同时验证：

- server 目录是否能标准 HF 加载
- embedding/head shape 是否与 baseline 保持一致
- logits / greedy / generated text 是否还能恢复

---

## 2. 结果文件

- 导出产物：
  - `artifacts/stage_i_phase2_square/server/`
  - `artifacts/stage_i_phase2_square/client/client_secret.pt`
- 回归结果：
  - `outputs/stage_i/phase2_square_regression.json`

---

## 3. shape 结果

`outputs/stage_i/phase2_square_regression.json` 中的 `shape_summary` 显示：

- `server_load_success = true`
- `embed_shape_match_baseline = true`
- `lm_head_shape_match_baseline = true`

具体 shape：

- `candidate_embed_shape = [151936, 896]`
- `baseline_embed_shape = [151936, 896]`
- `candidate_lm_head_shape = [151936, 896]`
- `baseline_lm_head_shape = [151936, 896]`

这说明：

> **square monomial transform 的 embed/head-only 版本，已经成功满足“标准 HF checkpoint shape 不变”这个目标。**

这是与当前扩维 KeyMat 原型最本质的区别。

---

## 4. transform 本身结果

同一份结果文件里，`square_transform` 的数值检查为：

- `max_abs_error = 0.0`
- `mean_abs_error = 0.0`
- `passes_tolerance = true`
- `global_scale = 1.0`

说明：

> 这一版 square transform 在数学上是严格可逆的，没有数值逆性问题。

---

## 5. 功能结果

### 5.1 汇总指标

在 `outputs/stage_i/phase2_square_regression.json` 中，汇总结果是：

- `avg_full_logits_max_abs_error = 34.6061`
- `avg_full_logits_mean_abs_error = 4.3688`
- `avg_last_token_logits_max_abs_error = 21.4515`
- `avg_last_token_logits_mean_abs_error = 3.7196`
- `greedy_first_token_match_rate = 0.0`
- `generated_ids_exact_match_rate = 0.0`
- `generated_text_exact_match_rate = 0.0`
- `candidate_has_nan_or_inf = false`

### 5.2 文本现象

baseline 仍是正常句子，例如：

- `我是一个人工智能助手，可以帮助您解答`
- `Machine learning allows computers to learn from data`

而 candidate 的输出已经变成明显错误的文本碎片，例如：

- `BöyleonomyPECPECshipPECPECship`
- `arestitectonomicPECshiponomicimatship`

这说明：

> **只改 embed/head，而不继续把 hidden transform 融到中间 attention / FFN / norm / residual 中，功能不会成立。**

---

## 6. 这一步到底证明了什么

这一步的价值不在于“功能已经成功”，而在于它同时证明了两件事：

### 6.1 正面结论

非扩维 square transform 确实能做到：

- checkpoint shape 不变
- 标准 HF 导出/加载不出问题

这为后续继续做：

- attention
- FFN
- norm

的标准权重物化，提供了一个**正确的结构起点**。

### 6.2 负面结论

仅做 embed/head-only 是**不够的**。

也就是说，Phase 2 的下一步不能停留在：

- “再微调一下噪声”
- 或“换个 square transform seed”

因为当前失败的原因不是随机参数点不好，而是：

> **hidden basis 已经变了，但中间层完全没有协变适配。**

---

## 7. 当前结论

本轮最小原型可以正式判定为：

### 已完成

- square monomial hidden transform：完成
- embed/head-only 标准 checkpoint 导出：完成
- HF shape 验证：完成
- HF 功能回归：完成

### 未完成

- 功能正确性：**未完成**

而且当前结果明确显示：

> 失败不是因为 shape，不是因为导出，不是因为数值逆性，而是因为 **attention / FFN / norm / residual 还没有接上这套新的 non-expanding hidden transform**。

---

## 8. 下一步最小实现切口

既然 embed/head-only 已经证明“不够”，那么最自然的下一步就是：

1. 保持当前 square monomial hidden transform
2. 先只做 **block0**
3. 顺序建议：
   - `input norm`
   - `attention`
   - `post-attn norm`
   - `FFN`
4. 先看：
   - `layer_0_attn_out`
   - `layer_0_block_out`
   - `final_logits`

也就是说，后面的最小可行目标不再是“更漂亮的 embed/head-only”，而是：

> **把 non-expanding hidden transform 从 embed/head-only 推进到 block0。**

---

## 9. 一句话结论

> 这个最小原型已经证明：**non-expanding / shape-preserving 路线在结构上是对的，但 embed/head-only 在功能上一定不够；Phase 2 真正的下一步必须是把 square hidden transform 继续推进到 block0 的 attention / FFN / norm。**
