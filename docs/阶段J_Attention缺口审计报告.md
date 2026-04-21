# 阶段 J：Attention 缺口审计报告

## 1. 目的

这份报告用于回答：

> 当前 buffered redesign 里的 attention 附加结构，是否已经被 standard-visible bridge 显式 materialize。

## 2. 当前输出

- `outputs/stage_j/attention_gap_report.json`

## 3. 解读方式

如果以下对象仍然显著偏离 identity：

- `q_feature_inv_order`
- `kv_feature_inv_order`
- `q_dense_inverse`
- `k_dense_inverse`

则说明：

- 当前 bridge 虽然已经写出了标准 `qkv/o` 权重
- 但 buffered redesign 里的 attention 内部结构信息仍未被 export-visible 化

这时，后续主阻塞就会继续从 `norm` 转移到 `attention`。 

## 4. 当前结果

当前审计结果是：

- `q_order_is_identity = false`
- `kv_order_is_identity = false`
- `q_dense_identity_max_abs_error ≈ 2.035`
- `k_dense_identity_max_abs_error ≈ 2.018`

因此当前最准确的结论是：

> bridge 线虽然已经写出了标准 `qkv/o` 权重，但 buffered redesign 里的 attention 内部重排与 dense 结构仍然没有被 export-visible 化。 
