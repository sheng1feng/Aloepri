# 阶段 J：KeyMat 候选搜索报告

## 1. 目的

当前论文一致部署的最大结构性阻塞，是：

- buffered redesign 的 norm metric 是强 full-metric 结构
- 标准 RMSNorm 导出与其存在强张力

因此本报告的目标不是直接修 bridge，而是先搜索：

> 是否存在更 “norm-friendly” 的 KeyMat 候选，使 `QQ^T` 更接近对角 / 标量结构。

## 2. 当前输出

- `outputs/stage_j/keymat_search.json`

## 3. 如何解读

最重要的指标是：

- `offdiag_ratio`

如果最优候选的 `offdiag_ratio` 仍然很高，就说明：

- 当前 Algorithm 1 参数族本身就天然倾向 full metric
- 后续更合理的方向可能不是“微调 bridge”，而是“重新约束 KeyMat 生成”
