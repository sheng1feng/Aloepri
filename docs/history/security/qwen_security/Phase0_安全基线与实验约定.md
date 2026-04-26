# Phase 0：安全基线与实验约定

## 1. 文档目的

本文件是 `Qwen` 安全评测工作的 **Phase 0 落地文档**。

它不直接实现攻击脚本，而是先把后续所有安全实验都必须统一的内容定下来：

- 威胁模型
- 评测对象
- 数据范围
- 输出 schema
- 目录规范
- 阶段工作拆分
- 验收标准

如果没有这一层约定，后续 `VMA / IA / IMA / ISA / TFMA / SDA` 很容易各自为战，最后难以横向对比。

---

## 2. Phase 0 目标

Phase 0 的目标不是“做出攻击结果”，而是完成以下四件事：

1. **把安全评测边界说清楚**
   - 哪些攻击算在当前轮；
   - 哪些攻击不算在当前轮；
   - 什么叫“部署态中间态攻击”；
   - 什么不属于真实部署威胁。

2. **把实验对象固定下来**
   - 哪些工件是首轮必须评测的；
   - 哪些 profile 是强制对比项；
   - 哪些数据集用于 smoke / train / eval。

3. **把输出格式统一下来**
   - 各攻击脚本必须输出什么字段；
   - summary、config、metrics、artifacts 怎么组织；
   - 后续如何自动聚合。

4. **把推进顺序固化下来**
   - Phase 1 先做什么；
   - Phase 1 与 Phase 3、Phase 4 的接口如何复用；
   - 何时可以判定“Phase 0 完成，可以进入攻击实现”。

---

## 3. 威胁模型基线

## 3.1 攻击者能力

当前轮次统一使用如下威胁模型：

- 服务端是 `honest-but-curious`
- 服务端可获得：
  - obfuscated model / release 工件
  - obfuscated 输入 token
  - obfuscated 输出 token / logits
- 在“部署态中间态攻击”设置下，服务端还可能接触：
  - attention score
  - decoder hidden states
  - layer outputs
  - KV cache
  - 推理日志中暴露的张量摘要

## 3.2 不计入攻击能力的内容

以下内容**不纳入**当前轮次攻击面：

- 本机开发中临时插入的 `hook`
- `recorder`
- 人工导出的 debug trace
- 仅调试环境才存在的中间态 dump

也就是说：

> 我们评估的是“真实 client/server 分离部署后，server 可能自然接触到的信息”，不是“开发者为了调试而额外暴露的信息”。

## 3.3 安全目标

当前轮次安全工作统一回答以下问题：

1. 服务端能否恢复秘密 token permutation；
2. 服务端能否从 obfuscated embedding / hidden representation 训练出反演器；
3. 服务端能否利用部署态中间量恢复输入 token；
4. 服务端能否通过长期 token 频率恢复文本分布或具体文本。

---

## 4. 首轮评测对象

## 4.1 工件对象

首轮统一评测以下工件：

- `artifacts/stage_h_full_obfuscated`
- `artifacts/stage_j_full_square`
- `artifacts/stage_j_full_square_tiny_a`
- `artifacts/stage_k_release`

这样做的原因：

- `Stage H` 代表更接近论文研究线；
- `Stage J` 代表标准形状 full-layer；
- `Stage K` 代表最终交付包装。

## 4.2 profile 对象

首轮强制对比：

- `stable_reference`
- `tiny_a`

可选扩展：

- `paper_default`
- 额外弱噪声点
- 额外强噪声点

## 4.3 数据对象

Phase 0 只定义数据类型，不要求马上下载全部数据。

### A. smoke 数据

用途：

- 快速跑通脚本接口；
- 验证输入输出 schema；
- 检查攻击脚本不会因为格式问题失败。

建议：

- 直接复用仓库已有 prompt；
- 再加一小份本地文本样本。

### B. inversion 训练数据

用途：

- `NN / IMA`
- 部分 `ISA`

要求：

- 有足够 token 覆盖度；
- 能切出 `train / val / test`；
- 最好为公开可复现文本。

### C. 频率攻击数据

用途：

- `TFMA`
- `SDA`

要求：

- 能构造三种设置：
  - 零知识
  - 域知识
  - 分布知识

---

## 5. 统一目录规范

Phase 0 建议统一后续安全实验目录如下：

### 文档

- `docs/qwen_security/`

### 脚本

- `scripts/security_qwen/`

建议后续按攻击分文件：

- `scripts/security_qwen/run_vma.py`
- `scripts/security_qwen/run_ia.py`
- `scripts/security_qwen/run_ima.py`
- `scripts/security_qwen/run_isa.py`
- `scripts/security_qwen/run_tfma.py`
- `scripts/security_qwen/run_sda.py`
- `scripts/security_qwen/export_security_matrix.py`

### 核心模块

- `src/security_qwen/`

建议后续拆分为：

- `src/security_qwen/schema.py`
- `src/security_qwen/datasets.py`
- `src/security_qwen/artifacts.py`
- `src/security_qwen/metrics.py`
- `src/security_qwen/vma.py`
- `src/security_qwen/ia.py`
- `src/security_qwen/ima.py`
- `src/security_qwen/isa.py`
- `src/security_qwen/tfma.py`
- `src/security_qwen/sda.py`

### 输出

- `outputs/security_qwen/`

建议按攻击类型分目录：

- `outputs/security_qwen/vma/`
- `outputs/security_qwen/ia/`
- `outputs/security_qwen/ima/`
- `outputs/security_qwen/isa/`
- `outputs/security_qwen/tfma/`
- `outputs/security_qwen/sda/`
- `outputs/security_qwen/summary/`

---

## 6. 统一结果 schema

所有攻击脚本必须输出统一 JSON 结构。

最少应包含四层：

### 6.1 顶层结构

```json
{
  "format": "qwen_security_eval_v1",
  "attack": "vma",
  "target": {...},
  "config": {...},
  "metrics": {...},
  "summary": {...},
  "artifacts": {...}
}
```

## 6.2 `target`

必须至少包含：

- `stage`
- `artifact_dir`
- `profile`
- `model_family`
- `variant`

## 6.3 `config`

必须至少包含：

- 攻击名称
- 数据集名称
- 数据切分
- 关键超参数
- 随机种子
- 设备
- dtype

## 6.4 `metrics`

按攻击类型扩展，但命名要统一。

推荐保留的标准字段：

- `token_top1_recovery_rate`
- `token_top10_recovery_rate`
- `token_top100_recovery_rate`
- `sensitive_token_recovery_rate`
- `embedding_cosine_similarity`
- `bleu4`
- `attack_runtime_seconds`

对于 `ISA` 再额外保留：

- `observable_type`
- `observable_layer`
- `observable_shape_summary`
- `intermediate_top1_recovery_rate`

## 6.5 `summary`

必须是“方便横向汇总”的短字段。

最少应包含：

- `status`
- `primary_metric_name`
- `primary_metric_value`
- `risk_level`
- `notes`

## 6.6 `artifacts`

用于记录辅助产物路径：

- 模型 checkpoint
- 中间缓存
- 排名表
- 错例样本
- 图表路径

---

## 7. 统一评测矩阵

Phase 0 要先把矩阵定义好，后面才能逐格填结果。

## 7.1 静态权重攻击矩阵

攻击：

- `VMA`
- `IA`

对象：

- `Stage H`
- `Stage J stable_reference`
- `Stage J tiny_a`
- `Stage K stable_reference`
- `Stage K tiny_a`

## 7.2 训练式反演矩阵

攻击：

- `NN`
- `IMA`

对象：

- `Stage A`
- `Stage H`
- `Stage J stable_reference`
- `Stage J tiny_a`

## 7.3 部署态中间态攻击矩阵

攻击：

- `ISA`

可见中间量：

- `attention_score`
- `hidden_state`
- `layer_output`
- `kv_cache`

对象：

- `Stage H`
- `Stage J`
- `Stage K`

## 7.4 在线统计攻击矩阵

攻击：

- `TFMA`
- `SDA`

知识设置：

- `zero_knowledge`
- `domain_aware`
- `distribution_aware`

对象：

- `Stage J stable_reference`
- `Stage J tiny_a`
- `Stage K stable_reference`
- `Stage K tiny_a`

---

## 8. Phase 0 详细工作计划

## Workstream A：威胁模型与对象冻结

任务：

1. 明确每类攻击的攻击输入；
2. 明确每类攻击的攻击输出；
3. 明确哪些攻击是“论文复现”，哪些是“工程扩展”；
4. 冻结首轮工件、profile、数据类型。

交付：

- 本文档定稿
- 首轮工件与 profile 清单

验收标准：

- 团队内任何人都能回答“这个攻击到底允许看到什么”

## Workstream B：目录与 schema 冻结

任务：

1. 冻结脚本目录；
2. 冻结输出目录；
3. 冻结 JSON schema；
4. 明确汇总脚本所需字段。

交付：

- 统一目录规范
- 统一结果 schema

验收标准：

- 后续攻击脚本不需要再各自定义输出格式

## Workstream C：评测矩阵冻结

任务：

1. 把研究线与部署线拆开；
2. 为每种攻击定义首轮必跑对象；
3. 明确哪些格子是 `must-have`，哪些是 `nice-to-have`。

交付：

- 首版评测矩阵

验收标准：

- 能据此直接生成任务列表，不再临时拍脑袋选实验点

## Workstream D：验收口径冻结

任务：

1. 为每个 Phase 1+ 攻击定义最小完成标志；
2. 为首版安全报告定义必须包含的图表/表格；
3. 为“风险高/中/低”定义统一口径。

交付：

- 攻击完成判据
- 首版报告验收口径

验收标准：

- 后续不会出现“跑了很多实验，但无法判断是否完成”的问题

---

## 9. 验收标准

Phase 0 完成必须同时满足以下条件：

### A. 文档层

- [ ] `Phase0_安全基线与实验约定.md` 定稿
- [ ] `README.md` 与 `推进看板.md` 已指向最新边界
- [ ] In-scope / Out-of-scope 无歧义

### B. 对象层

- [ ] 首轮工件清单已冻结
- [ ] 首轮 profile 清单已冻结
- [ ] 数据类型与切分方案已冻结

### C. 结构层

- [ ] 安全实验目录规范已冻结
- [ ] 攻击输出 schema 已冻结
- [ ] 汇总字段已冻结

### D. 管理层

- [ ] Phase 1 / 3 / 4 / 5 的输入输出依赖已明确
- [ ] 首版评测矩阵已明确
- [ ] 首版安全报告最小内容已明确

## 10. 不通过条件

出现以下任一情况，Phase 0 都不应视为完成：

- 不同攻击脚本仍准备输出不同 JSON 结构；
- `Stage H / J / K` 的对象边界还在摇摆；
- `ISA` 还没定义“到底什么中间量算可见”；
- 频率攻击的数据设置还没明确；
- 后续阶段还无法据此直接拆任务。

---

## 11. Phase 0 完成后的直接下一步

完成 Phase 0 后，应立刻进入以下三个实现方向：

1. `Phase 1 / VMA`
   - 先做最直接的 permutation 恢复基线
2. `Phase 3 / NN-IMA`
   - 先做训练式 embedding 反演
3. `Phase 4 / ISA`
   - 先定义部署态可见中间量接口，再做最小攻击闭环

原因：

- 这三类攻击分别对应：
  - 静态权重侧
  - 训练式反演侧
  - 部署态中间态侧

它们组合起来，最早就能给出一版有代表性的安全基线。

---

## 12. 一句话结论

> Phase 0 的任务不是“先跑攻击”，而是先把 `Qwen` 安全评测的边界、对象、目录、schema、矩阵和验收口径一次性定清楚，使后续每一种攻击都能在同一套规则下推进、对比和收敛。
