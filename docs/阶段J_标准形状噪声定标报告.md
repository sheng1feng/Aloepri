# 阶段 J：标准形状 full-layer 噪声定标报告

> Legacy note: 本文档对应的是 **旧版 Stage J** 的 conservative standard-shape 噪声定标。新版 `Stage J` 的 canonical 定义见 `docs/阶段J_Qwen全模型部署物化报告.md`。

本文档记录在 **standard-shape / non-expanding full-layer** 路线上的第一轮噪声定标结果。

本轮的目标不是隐私攻击评估，而是回答：

> 在已经打通 `stage_j_full_square` 的前提下，`alpha_e / alpha_h` 应该怎么设，既能保持 full-layer 标准形状模型的功能正确性，又能引入非零噪声。

---

## 1. 背景

当前阶段 J 已经完成：

- `prefix-1 / 2 / 4 / 8`
- `full-layer`
- `full-layer` 标准 HF checkpoint 导出

并且在零噪声下实现了近乎精确恢复：

- `avg_full_logits_max_abs_error ≈ 1.26e-4`
- `generated_ids_exact_match_rate = 1.0`

因此现在可以开始问一个更实际的问题：

> **在 standard-shape 路线上，非零噪声还能保留多少功能？**

---

## 2. 本轮实验设置

脚本：

- `scripts/run_stage_j_noise_calibration.py`

结果文件：

- `outputs/stage_j/noise_calibration.json`

固定设置：

- 模型：`Qwen2.5-0.5B-Instruct`
- 路线：`stage_j_full_square`
- hidden transform：square monomial
- `global_scale = 1.0`
- `dtype = float32`
- prompts：默认 5 条固定 prompt

测试噪声点：

- `stable_reference`：`alpha_e=0.0, alpha_h=0.0`
- `tiny_a`：`alpha_e=0.02, alpha_h=0.01`
- `tiny_b`：`alpha_e=0.05, alpha_h=0.02`
- `small_a`：`alpha_e=0.1, alpha_h=0.05`
- `small_b`：`alpha_e=0.15, alpha_h=0.05`
- `small_c`：`alpha_e=0.1, alpha_h=0.1`
- `paper_like`：`alpha_e=1.0, alpha_h=0.2`

---

## 3. 排名结果

按当前排序规则（先看生成，再看 logits / block_out）：

1. `stable_reference`
2. `tiny_a`
3. `tiny_b`
4. `small_a`
5. `small_b`
6. `small_c`
7. `paper_like`

也就是说：

> **当前 standard-shape full-layer 路线的最佳非零工作点是 `tiny_a`。**

---

## 4. 关键结果

### 4.1 零噪声参考点

`stable_reference`

- `avg_final_logits_restored_max_abs_error ≈ 1.26e-4`
- `avg_layer_23_block_out_max_abs_error ≈ 5.80e-5`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

解释：

- 这是当前 standard-shape full-layer 的理想基线；
- 后续所有非零噪声点都应和它比较。

### 4.2 推荐非零工作点

`tiny_a = (alpha_e=0.02, alpha_h=0.01)`

- `avg_final_logits_restored_max_abs_error ≈ 0.672`
- `avg_layer_23_block_out_max_abs_error ≈ 0.304`
- `avg_final_hidden_max_abs_error ≈ 1.620`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

解释：

- 这是当前最好的 **非零噪声工作点**
- 虽然数值误差开始明显高于零噪声基线，但生成仍保持完全一致

### 4.3 次优非零工作点

`tiny_b = (alpha_e=0.05, alpha_h=0.02)`

- `avg_final_logits_restored_max_abs_error ≈ 1.687`
- `avg_layer_23_block_out_max_abs_error ≈ 0.801`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

解释：

- 仍然保持生成功能完全一致；
- 但误差明显大于 `tiny_a`

### 4.4 中等噪声区间

`small_a = (0.1, 0.05)`

- `avg_final_logits_restored_max_abs_error ≈ 3.441`
- `generated_ids_exact_match_rate = 0.6`
- `generated_text_exact_match_rate = 0.6`

`small_b = (0.15, 0.05)`

- `avg_final_logits_restored_max_abs_error ≈ 4.913`
- `generated_ids_exact_match_rate = 0.6`
- `generated_text_exact_match_rate = 0.6`

`small_c = (0.1, 0.1)`

- `avg_final_logits_restored_max_abs_error ≈ 3.664`
- `generated_ids_exact_match_rate = 0.2`
- `generated_text_exact_match_rate = 0.2`

解释：

- 到这一档后，生成已经开始明显退化；
- 仍然不是“马上乱码”，但已经不适合作为默认推荐点

### 4.5 paper-like 点

`paper_like = (1.0, 0.2)`

- `avg_final_logits_restored_max_abs_error ≈ 17.03`
- `avg_layer_23_block_out_max_abs_error ≈ 24.21`
- `avg_final_hidden_max_abs_error ≈ 117.21`
- `greedy_first_token_match_rate = 0.8`
- `generated_ids_exact_match_rate = 0.0`
- `generated_text_exact_match_rate = 0.0`

解释：

- 这说明论文量级的噪声在当前 `Qwen2.5-0.5B` + standard-shape 路线下明显过强；
- 它不能直接作为当前实现的默认工作点

---

## 5. 推荐配置

### 配置 A：零噪声研究基线

- `alpha_e = 0.0`
- `alpha_h = 0.0`

适用：

- 继续做功能正确性验证
- 继续做 HF / vLLM 路径排障

### 配置 B：推荐非零工作点

- `alpha_e = 0.02`
- `alpha_h = 0.01`

适用：

- 需要引入非零噪声
- 但又希望当前 full-layer 生成功能保持不变

---

## 6. 导出工件

本轮还额外导出了一版推荐非零工作点的 full-layer 工件：

- `artifacts/stage_j_full_square_tiny_a/server/`
- `artifacts/stage_j_full_square_tiny_a/client/client_secret.pt`

并做了导出后 HF 回归：

- `outputs/stage_j/full_layers_square_tiny_a_export_regression.json`

汇总结果：

- `avg_full_logits_max_abs_error ≈ 0.672`
- `avg_last_token_logits_max_abs_error ≈ 0.184`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

这说明：

> `tiny_a` 不仅在内存里是当前最佳非零点，而且导出成标准 checkpoint 之后仍保持同级别可用性。

---

## 7. 当前结论

本轮阶段 J 噪声定标可以明确给出三个结论：

1. `stage_j_full_square` 已经具备做噪声定标的稳定性基础
2. 当前最佳非零工作点是：
   - `alpha_e = 0.02`
   - `alpha_h = 0.01`
3. 论文量级噪声在当前小模型和当前 standard-shape 实现下明显过强

---

## 8. 一句话结论

> 在当前 standard-shape full-layer 路线上，**推荐非零工作点是 `tiny_a = (alpha_e=0.02, alpha_h=0.01)`**；它已经在 full-layer 和导出后 HF 回归中保持 `generated_ids/text exact match = 1.0`，可以作为后续继续推进的默认噪声配置。
