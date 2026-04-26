# 阶段 J：标准形状协变恢复计划

本文档重新定义阶段 J，使其直接承接当前阶段 I 的真实卡点。

当前仓库的事实已经很清楚：

1. **阶段 I / Phase 1**（Stage A 标准 HF/vLLM 导出）已经完成；
2. 当前扩维 KeyMat 无法继续 materialize 到标准 HF/vLLM checkpoint；
3. 新的 **non-expanding / shape-preserving** 最小原型已经证明：
   - shape 可以保持标准；
   - 但只做 `embed/head-only` 时，功能完全不成立。

因此，阶段 J 不应该再定义成“泛泛的下一步”，而应该精准定义为：

> **把 non-expanding hidden transform 从 `embed/head-only` 推进到 `block0 -> prefix -> full-layer`，验证它能否在保持标准 checkpoint shape 的前提下恢复功能。**

---

## 1. 当前卡点到底是什么

当前不是卡在导出，也不是卡在 token mapping，而是卡在：

### 卡点 J-1：扩维 KeyMat 不能进入标准 checkpoint

`outputs/stage_i/phase2_probe.json` 已经证明：

- baseline hidden size：`896`
- KeyMat expanded size：`1152`

因此：

- embedding/head shape 变了
- q/k/v/o shape 变了
- gate/up/down shape 变了
- RMSNorm 也不再是标准 weight 向量语义

这意味着当前扩维 KeyMat 不能直接作为标准 HF/vLLM Phase 2 的继续方向。

### 卡点 J-2：non-expanding 原型只做 embed/head 不够

`outputs/stage_i/phase2_square_regression.json` 已经证明：

- `embed_shape_match_baseline = true`
- `lm_head_shape_match_baseline = true`

但同时：

- `avg_full_logits_max_abs_error ≈ 34.61`
- `greedy_first_token_match_rate = 0.0`
- `generated_ids_exact_match_rate = 0.0`

说明：

> 只把 hidden transform 吃进 embed/head，而不继续让中间层协变适配，系统功能必然崩。

所以阶段 J 的真正任务就是：

> **把 non-expanding hidden transform 继续推进到中间层。**

---

## 2. 阶段 J 的目标

阶段 J 的目标不是部署，也不是攻击评估，而是：

> **在保持原始 Qwen2 标准 checkpoint shape 不变的前提下，把 non-expanding hidden transform 逐步扩展到中间层，使模型重新恢复功能正确性。**

具体来说，阶段 J 要回答三个问题：

1. `block0` 能不能在标准 shape 下恢复？
2. `prefix-2 / prefix-4` 能不能在标准 shape 下继续传播？
3. 如果能，是否可以继续把这种形态导出为更接近 vLLM 的标准 HF checkpoint？

---

## 3. 阶段 J 的范围

### In scope

- non-expanding square hidden transform
- block0 / prefix-2 / prefix-4 的协变恢复
- 标准 checkpoint shape 下的功能回归
- 基于 HF 的 correctness 验证

### Out of scope

- 安全攻击评估（VMA / IA / ISA / IMA）
- 更通用的导出包装
- CLI/API 产品化
- `src/aloepri/` 模块化抽象

---

## 4. 技术路线

### J1：先做 block0 full recovery（标准 shape 版本）

顺序固定为：

1. `input norm`
2. `attention`
3. `post-attn norm`
4. `FFN`

这一步的目标不是 full-layer，而是先验证：

\[
Block_0(xP) \approx Block_0(x)P
\]

同时所有参数 shape 仍保持与原始 Qwen2 一致。

#### 验收

- `layer_0_attn_out` 误差显著下降
- `layer_0_block_out` 明显恢复
- `final_logits` 至少开始回归

### J2：prefix-2

在 block0 成立后，再做：

- `layer_0`
- `layer_1`

目标：

- 验证非扩维 transform 在两层传播下是否仍可控

#### 验收

- `layer_1_block_out` 不出现断崖爆炸
- greedy / generated ids 开始有恢复趋势

### J3：prefix-4

如果 prefix-2 稳定，再推进到：

- `layer_0..3`

目标：

- 看误差是缓慢增长还是快速发散

#### 验收

- 不出现“shape 正确但功能完全失控”
- 误差曲线可解释

### J4：标准 checkpoint 再导出

如果 prefix-4 稳定，再尝试导出一版新的标准 checkpoint：

- 仍保持原始 shape
- 但已经不仅是 embed/head-only
- 至少包含 block0 或 prefix-2 的 materialized 结果

这一步是为下一轮重新接 vLLM 做准备。

---

## 5. 推荐的实现顺序

### 第 1 步：block0 norm

先只解决：

- `input_layernorm`
- `post_attention_layernorm`

前提是 transform family 仍限定在：

- permutation
- sign
- 可选全局 scale

### 第 2 步：block0 attention

把当前阶段 E/H 中稳定的 attention 复杂结构迁移到 non-expanding hidden transform 下。

注意：

- 不是重新发明 attention profile
- 而是让现有 `R̂_qk / Ĥ_qk / Ẑ_block / τ_kv / τ_group` 在标准 shape 下工作

### 第 3 步：block0 FFN

保留已有：

- `Z_ffn`
- `H_ffn`

只替换 hidden 输入输出侧的 transform。

### 第 4 步：prefix-2 / prefix-4

先扩到两层，再扩到四层，不要一开始 full-layer。

---

## 6. 阶段 J 与阶段 I / K 的区别

### 阶段 I

阶段 I 的核心问题是：

> **能不能把当前混淆模型物化成标准 HF/vLLM checkpoint，并定义清楚 client/server 在线契约。**

它关心的是：

- 导出格式
- token/logit 映射
- 标准 checkpoint
- vLLM/HF 加载

阶段 I 不负责把一个功能不成立的 non-expanding transform 自动变成可用模型。

### 阶段 J

阶段 J 的核心问题是：

> **在不改变标准 checkpoint shape 的前提下，如何让 non-expanding hidden transform 在中间层也功能成立。**

它关心的是：

- block0 / prefix 的协变恢复
- norm / attention / FFN 的标准 shape 适配
- 功能正确性

### 阶段 K

阶段 K 的核心问题应定义为：

> **当 I + J 已经把“标准 shape + 功能正确”打通后，如何把它更完整地包装成可交付、可分发、可服务化的部署形态。**

它关心的是：

- 更通用的导出格式
- baseline-free loader
- 更接近原生 `save_pretrained` / server API 的交付体验

简化成一句话：

- **阶段 I**：先把标准导出和部署契约打通
- **阶段 J**：把 non-expanding 方案真的做成功能正确
- **阶段 K**：把已经可用的标准形态做成更完整的交付系统

---

## 7. 阶段 J 的验收标准

阶段 J 可以认为完成，当满足以下 4 条：

1. non-expanding transform 下，`block0` 的 `block_out` 恢复成立
2. prefix-2 不出现断崖爆炸
3. prefix-4 误差传播仍可解释
4. 至少能导出一版“比 embed/head-only 更进一步”的标准 checkpoint，并在 HF 路径看到恢复趋势

---

## 8. 一句话结论

> 阶段 J 的本质不是“继续导出”，而是**先把 non-expanding / 标准 shape 方案在 block0 到 prefix 多层上做成功能正确性恢复**；只有这一步成立，阶段 I 的 Phase 2 才能真正进入标准 HF/vLLM 主线。

---

## 9. 当前进展（第一轮实现后）

本计划的第一轮实现已经得到一组正面结果：

- `prefix-1`：`outputs/stage_j/prefix_1_square.json`
- `prefix-2`：`outputs/stage_j/prefix_2_square.json`
- `prefix-4`：`outputs/stage_j/prefix_4_square.json`
- `prefix-8`：`outputs/stage_j/prefix_8_square.json`
- `full-layer`：`outputs/stage_j/full_layers_square.json`
- `full-layer export regression`：`outputs/stage_j/full_layers_square_export_regression.json`

当前已经验证：

- `prefix-1`、`prefix-2`、`prefix-4` 都能在 HF 路径下恢复到近乎精确正确
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

当前已经验证：

- `prefix-1 / 2 / 4 / 8 / full-layer` 都能在 HF 路径下恢复到近乎精确正确
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`
- full-layer 版本已经可以导出为标准 HF checkpoint，并在重新加载后保持同级别回归结果

对应结果报告见：

- `docs/阶段J_标准形状前缀恢复报告.md`
