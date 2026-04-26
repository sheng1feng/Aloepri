# Gate 5：参数-安全性联合扫描计划

## 1. 目标

Gate 5 的目标是把前面 4 个 Gate 的攻击结果，与 `Stage J` 的参数工作点结合起来，回答：

1. `alpha_e / alpha_h` 上升时，accuracy 与 security 如何同时变化；
2. 当前最合理的推荐工作点是什么；
3. 哪些攻击会随参数变化而明显变化，哪些几乎不变。

本轮重点不是“再发明新的攻击”，而是把：

- `Gate 1 / VMA`
- `Gate 2 / IMA`
- `Gate 3 / ISA`
- `Gate 4 / TFMA + SDA`

统一投影到同一张参数-安全性表上。

---

## 2. 扫描对象

本轮聚焦 `Stage J` 的代表性参数点：

- `stable_reference`
- `tiny_a`
- `tiny_b`
- `small_a`
- `paper_like`

这样能覆盖：

- 零噪声正确性基线
- 推荐非零工作点
- 更强但仍保持正确性的点
- 已出现明显精度退化的点
- 论文级强噪声点

---

## 3. 评估指标

### Accuracy

- `generated_ids_exact_match_rate`
- `generated_text_exact_match_rate`
- `avg_final_logits_restored_max_abs_error`

### Security

- `vma_projection_top1`
- `ima_top1`
- `isa_hidden_top1`
- `tfma_domain_top10`
- `sda_distribution_bleu4`

其中：

- `TFMA / SDA` 预期主要由 token permutation 决定，可能对 `alpha_e / alpha_h` 不敏感；
- 若结果确实不变，这本身就是重要结论。

---

## 4. Gate 5 完成标准

- [ ] 至少 5 个参数点有统一结果
- [ ] 同一张表上同时包含 accuracy 与 security 指标
- [ ] 有推荐工作点结论
- [ ] 有 Gate 5 文档沉淀

---

## 5. 一句话结论

> Gate 5 的闭环重点是把“参数工作点选择”从 correctness-only 变成真正的 accuracy-privacy tradeoff 选择。 
