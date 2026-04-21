# 阶段 J：KeyMat 网格搜索报告

## 1. 目的

前一轮 seed 搜索已经表明：

- `offdiag_ratio` 对 seed 不敏感

所以这份报告继续回答：

> 问题到底是 seed，还是当前 `expansion_size / λ` 参数族本身。

## 2. 当前输出

- `outputs/stage_j/keymat_grid_search.json`

## 3. 如何解读

如果整张网格里最优 `best_offdiag_ratio` 仍然接近 1，那么说明：

- 当前参数族天然倾向 full-metric norm
- 后续应考虑改变 KeyMat 生成约束，而不是只靠桥接微调

## 4. 当前结果

当前搜索范围：

- `expansion_size ∈ {32, 64, 128, 256}`
- `λ ∈ {0.0, 0.1, 0.3, 1.0}`

当前最优候选是：

- `expansion_size = 256`
- `λ = 0.3`

对应：

- `best_offdiag_ratio ≈ 0.958`

相较当前默认 `h=128, λ=0.3` 的 `~0.972`，说明：

- 更大的 `h` 确实能把 `QQ^T` 稍微往更“norm-friendly”的方向推
- 但仍然远没到可被标准 RMSNorm 等价承接的程度

## 5. 对部署线的含义

我还用这组最优候选做了一条真实实验线，结果是：

- `norm_gap` 确实略有下降
- `bridge_regression` 的 `avg_restored_full_logits_max_abs_error` 也略有下降
- 但 `generated_ids_exact_match_rate` 仍然是 `0.0`

因此当前最准确的结论是：

> **当前 KeyMat 参数族在更大 `h` 下可以变得稍微更适合部署，但这还不足以直接解决论文一致部署所需的 norm/export 语义冲突。**

## 6. 与 `diag_friendly` family 的对比

后续我又新增了一条 `diag_friendly` family 实验线，并把它真正拉成了：

- `stage_h_pretrained_diagfriendly`
- `stage_j_qwen_redesign_diagfriendly`
- `stage_j_qwen_redesign_standard_diagfriendly`

这条线给出的结论非常关键：

- `norm_gap_summary.all_standard_rmsnorm_equivalent = true`
- 但 `bridge_summary.generated_ids_exact_match_rate = 0.0`

这说明：

> 只要改掉 KeyMat family，确实可以把 `norm` 这块从“结构性冲突”变成“可标准承接”；  
> 但这并不会自动让整条部署线等价，后续主阻塞会继续转移到 attention / FFN 侧表达。 
