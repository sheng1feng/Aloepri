# Gate 2：IMA 闭环计划

> 废弃说明：本文档对应旧的最小 ridge `IMA` baseline 计划，现已废弃为历史材料，不再代表当前 Qwen 主线的 `IMA` 口径。当前活跃结论只认 `paper_like IMA`。

## 1. 目标

Gate 2 的目标是完成一条**训练式反演攻击**闭环，用来回答：

1. 相比 `VMA` 的结构恢复，训练式 embedding 反演是否同样危险；
2. `Stage A`、`Stage H`、`Stage J / K` 在训练式反演下的风险排序是否改变；
3. 当前部署线的风险是“只对结构型攻击脆弱”，还是“对学习型反演也脆弱”。

一句话理解：

> Gate 1 证明了“结构能被恢复”；Gate 2 要证明“语义表示能否被学回来”。

---

## 2. 闭环要求

Gate 2 必须同时留下四类证据：

1. **代码证据**
   - `src/security_qwen/ima.py`
   - `scripts/security_qwen/run_ima.py`
   - `scripts/security_qwen/export_ima_comparison.py`

2. **测试证据**
   - `tests/test_security_qwen_ima*.py`

3. **结果证据**
   - `outputs/security_qwen/ima/*.json`
   - `outputs/security_qwen/summary/ima_comparison.json`

4. **文档证据**
   - `docs/qwen_security/Gate2_IMA结果记录.md`
   - `推进看板` 更新

---

## 3. 计划范围

### In scope

- 以 token-level obfuscated embedding 为输入
- 以 plaintext embedding / plaintext token id 为恢复目标
- 做最小“训练式反演器”闭环
- 横向比较：
  - `Stage A`
  - `Stage H`
  - `Stage J stable_reference`
  - `Stage J tiny_a`
  - 可选：`Stage K stable_reference`
  - 可选：`Stage K tiny_a`

### Out of scope

- 中间态反演（这属于 `Gate 3 / ISA`）
- 文本级 seq2seq 反演器
- 大规模外部公开语料下载
- GPU 大模型训练

---

## 4. 最小攻击方案

本轮使用一个**最小但可验证**的 IMA：

- 输入：obfuscated embedding 向量
- 输出：预测的 plaintext embedding 向量
- 恢复方式：
  1. 训练一个映射器 `f(z_obf) -> e_plain`
  2. 将预测结果与候选 plaintext embedding 做最近邻匹配
  3. 输出 Top-1 / Top-k token 恢复率

为了保证本轮可闭环、可快速验证，优先选择：

- **线性 / ridge inversion model**

原因：

- 训练成本低
- 可在 CPU 上快速跑通
- 足以作为第一版训练式反演基线

如果这一版已经能明显恢复 `Stage J / K`，就足以支撑 Gate 2 的阶段结论。

---

## 5. 数据与切分

本轮不下载新数据集，直接基于 token-row 级样本做最小闭环。

### 5.1 样本定义

对每个 plaintext token id：

- 输入样本：对应 obfuscated embedding row
- 标签：
  - plaintext embedding row
  - plaintext token id

### 5.2 切分方式

- `train_plain_ids`
- `val_plain_ids`
- `test_plain_ids`

并保证：

- test 集中尽量覆盖 prompt 中出现的敏感 token；
- 检索候选池中包含 test 真值 token 与额外 distractor。

### 5.3 评测指标

- `token_top1_recovery_rate`
- `token_top10_recovery_rate`
- `embedding_cosine_similarity`
- `sensitive_token_recovery_rate`

---

## 6. 工作包

## Work Package 2.1：数据构造

任务：

- 实现 IMA 样本构造器
- 实现 token id 切分
- 实现 candidate pool 构造

交付：

- `src/security_qwen/ima.py` 中的数据构造函数

验收：

- 对任一 target 都能稳定构造 train / val / test

## Work Package 2.2：最小反演器

任务：

- 实现线性 / ridge inversion model
- 支持简单超参数选择
- 支持从 obfuscated embedding 回归到 plaintext embedding

交付：

- `run_ima_baseline(...)`

验收：

- 至少一个 target 上能稳定输出非空指标

## Work Package 2.3：横向比较

任务：

- 在多个 target 上跑 IMA
- 导出统一比较结果

交付：

- `scripts/security_qwen/export_ima_comparison.py`
- `outputs/security_qwen/summary/ima_comparison.json`

验收：

- 能直接回答各工件风险排序

## Work Package 2.4：文档收口

任务：

- 写 Gate 2 结果记录
- 更新推进看板

验收：

- Gate 2 的结论可独立阅读

---

## 7. Gate 2 验收标准

Gate 2 完成必须满足：

- [ ] `run_ima.py` 可真实执行
- [ ] 至少 4 个 target 有真实结果
- [ ] 输出符合统一 schema
- [ ] 有 `token_top1_recovery_rate`
- [ ] 有 `token_top10_recovery_rate`
- [ ] 有 `embedding_cosine_similarity`
- [ ] 有 baseline vs 混淆对比
- [ ] 有 Gate 2 文档沉淀

---

## 8. 通过后下一步

若 Gate 2 完成，下一步进入：

- `Gate 3 / ISA`

届时要重点回答：

- 结构恢复危险，还是中间态恢复更危险；
- 部署态下最危险的信息暴露点到底是什么。

---

## 9. 一句话结论

> Gate 2 的闭环重点不是训练一个复杂模型，而是先用最小训练式反演器稳定证明：当前各类工件的 obfuscated embedding 是否仍然足以被学习性地映射回 plaintext 语义空间。 
