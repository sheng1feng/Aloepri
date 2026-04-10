# Qwen 安全评测与安全工作

本目录用于专门记录基于 `Qwen` 主线的安全评测规划、推进状态与后续安全增强工作。

当前约束：

- 以 `Qwen` 为基线模型与主要复现对象；
- 优先评估**静态权重侧**与**在线 token 侧**攻击；
- 当前轮次**考虑客户端 / 服务端分离后的中间态攻击**，例如服务端可见 attention score、KV cache、层输出等部署态中间信息；
- 但**不考虑本机开发阶段特有的 hook / recorder / debug trace 暴露**；
- 先完成“论文攻击复现 + 当前仓库版本安全基线建立”，再进入“增强点验证”。

建议阅读顺序：

1. `安全评测总计划.md`
2. `Phase0_安全基线与实验约定.md`
3. `执行计划_闭环版.md`
4. `Gate1_VMA最小基线对比.md`
5. `Gate2_IMA闭环计划.md`
6. `Gate2_IMA结果记录.md`
7. `Gate3_ISA闭环计划.md`
8. `Gate3_ISA结果记录.md`
9. `Gate4_TFMA_SDA闭环计划.md`
10. `Gate4_TFMA_SDA结果记录.md`
11. `Gate5_参数安全性联合扫描计划.md`
12. `Gate5_参数安全性联合扫描结果.md`
13. `Gate6_增强点验证计划.md`
14. `Gate6_增强点验证结果.md`
15. `部署线弱于StageH的原因分析.md`
16. `推进看板.md`

目录说明：

- `安全评测总计划.md`：整体路线、阶段划分、指标、优先级
- `Phase0_安全基线与实验约定.md`：Phase 0 的详细工作计划、schema 与验收标准
- `执行计划_闭环版.md`：后续 Gate 级执行顺序、依赖、产出与验收关卡
- `Gate1_VMA最小基线对比.md`：Gate 1 当前已拿到的 baseline vs obfuscated 首版对比结果
- `Gate2_IMA闭环计划.md`：Gate 2 的闭环目标、工作包与验收标准
- `Gate2_IMA结果记录.md`：Gate 2 当前已完成的训练式反演结果与结论
- `Gate3_ISA闭环计划.md`：Gate 3 的部署态 observable 攻击闭环计划
- `Gate3_ISA结果记录.md`：Gate 3 当前已完成的 hidden_state / attention_score 结果与结论
- `Gate4_TFMA_SDA闭环计划.md`：Gate 4 的长期在线统计攻击闭环计划
- `Gate4_TFMA_SDA结果记录.md`：Gate 4 当前已完成的 TFMA / SDA 结果与结论
- `Gate5_参数安全性联合扫描计划.md`：Gate 5 的 tradeoff 扫描计划
- `Gate5_参数安全性联合扫描结果.md`：Gate 5 当前已完成的参数-安全性扫描结果与推荐工作点
- `Gate6_增强点验证计划.md`：Gate 6 的增强验证计划
- `Gate6_增强点验证结果.md`：Gate 6 当前已完成的增强点验证与结论
- `部署线弱于StageH的原因分析.md`：为什么标准部署线会显著弱于研究线的根因分析
- `推进看板.md`：阶段状态、里程碑、待办
