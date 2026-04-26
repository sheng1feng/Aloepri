# 阶段 J：Norm 缺口审计报告

## 1. 目的

这份报告用于把当前 `bridge` 与 `buffered redesign` 之间的 `norm` 差距量化出来。

目标不是给出最终修复，而是明确回答：

> 当前 buffered redesign 的 norm metric，是否本质上可以由标准 RMSNorm 的单个权重向量等价承接。

## 2. 当前输出

- `outputs/stage_j/norm_gap_report.json`

## 3. 当前判断

如果 `offdiag_ratio` 很高，那么说明：

- 当前 norm 不是“接近对角矩阵”
- 它更像一个 full metric 结构
- 单靠标准 RMSNorm 权重向量无法等价表达

这时，bridge 线里使用 `ones` 或其他简单启发式，就不是“暂时没实现完整逻辑”那么简单，而是：

> **当前部署约束与 buffered redesign norm 语义之间存在真实结构性张力。**

## 4. 当前策略扫描结果

当前 bridge norm 策略扫描结果显示：

- `ones`
  - `avg_restored_full_logits_max_abs_error ≈ 31.74`
  - `generated_ids_exact_match_rate = 0.0`
- `metric_diag_sqrt`
  - 明显更差
- `kappa_fused`
  - `avg_restored_full_logits_max_abs_error ≈ 28.41`
  - `generated_ids_exact_match_rate = 0.2`

这说明：

- 单纯用对角近似去拟合 full metric norm 是错误方向
- 论文式 `κ` 标量修正至少是当前 bridge 线上更合理的方向
- 但 `kappa_fused` 仍然远未达到等价程度，因此 `norm` 仍不能算彻底解决
