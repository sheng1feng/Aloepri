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
