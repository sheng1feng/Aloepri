# 阶段 J：桥接等价性回归报告

## 1. 目的

这份报告用于量化：

> `buffered redesign line` 与 `standard-visible bridge line` 在真实 prompt 上到底差多远。

它不回答“哪条线更安全”，而是回答：

- bridge 当前是否只是可加载；
- 还是已经在数值和生成上足够接近 buffered redesign。

## 2. 当前输出

- `outputs/stage_j/bridge_regression.json`

## 3. 当前结果

当前 bridge regression 汇总结果是：

- `avg_restored_full_logits_max_abs_error ≈ 31.74`
- `avg_restored_full_logits_mean_abs_error ≈ 2.52`
- `avg_restored_last_token_max_abs_error ≈ 23.46`
- `generated_ids_exact_match_rate = 0.0`
- `generated_text_exact_match_rate = 0.0`

这说明：

> 当前 bridge 线虽然已经是标准键可见、并且可被标准 Hugging Face 加载器加载，但它还远没有达到与 buffered redesign line 等价的程度。

## 4. 当前最合理的解释

当前 bridge 线已经做了：

- `embed / lm_head / qkv / o / gate / up / down` 的标准键映射
- `config` 的形状重写

但它还没有完成：

- attention 内部附加结构的等价 materialization
- `norm` 的真正等价桥接

因此当前最合理的判断是：

> 当前 bridge 线已完成“标准可见导出”，但尚未完成“语义等价导出”。 

## 5. 当前建议

后续若继续推进等价性，应优先检查：

1. `norm` 如何从 placeholder/近似桥接推进到更接近等价的实现
2. attention 内部附加结构是否需要进一步显式 materialize
3. bridge regression 是否能在下一轮明显下降
