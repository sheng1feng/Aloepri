# Gate 4：TFMA + SDA 闭环计划

## 1. 目标

Gate 4 的目标是完成一条**长期在线统计攻击**闭环，用来回答：

1. 仅通过长期观察 obfuscated token 频率，攻击者能恢复多少 plaintext token；
2. 在引入序列共现 / bigram 结构后，攻击是否会显著增强；
3. `Stage A/H/J/K` 在频率攻击下是否仍存在可区分差异；
4. 四条攻击线（`VMA / IMA / ISA / TFMA+SDA`）里，长期统计攻击到底排在什么风险级别。

一句话理解：

> Gate 4 关注的不是“权重”或“中间态”，而是“服务端长期看见的 obfuscated token 序列统计本身”。

---

## 2. 本轮范围

### In scope

- `TFMA`
- `SDA`
- 三种知识设置：
  - `zero_knowledge`
  - `domain_aware`
  - `distribution_aware`
- 对比对象：
  - `stage_a_standard`
  - `stage_h_full_obfuscated`
  - `stage_j_stable_reference`
  - `stage_j_tiny_a`
  - `stage_k_stable_reference`
  - `stage_k_tiny_a`

### Out of scope

- 真实外部公开数据集下载
- 复杂神经 decipher 模型
- 长上下文对话级统计攻击

本轮优先使用**本地可重复生成的文本语料**闭环。

---

## 3. 最小攻击方案

### 3.1 TFMA

最小 `TFMA`：

- 统计 reference plaintext corpus 的 token 频率
- 统计 private obfuscated corpus 的 token 频率
- 用频率排序 / 频率邻近匹配恢复 token
- 输出：
  - `token_top1_recovery_rate`
  - `token_top10_recovery_rate`
  - `token_top100_recovery_rate`
  - `sensitive_token_recovery_rate`

### 3.2 SDA

最小 `SDA`：

- 在 `TFMA` 的基础上引入 token bigram 共现结构
- 为每个 token 构造：
  - unigram 频率
  - sorted outgoing bigram profile
  - sorted incoming bigram profile
- 做 substitution-style matching
- 再恢复 private corpus 中的 token 序列
- 输出：
  - `token_top1_recovery_rate`
  - `token_top10_recovery_rate`
  - `token_top100_recovery_rate`
  - `bleu4`

---

## 4. 数据设置

本轮不下载外部数据，而是用本地可复现的文本生成器构造三种知识设置。

### 4.1 `zero_knowledge`

- reference：通用主题文本
- private：隐私 / 模型 / 推理主题文本

### 4.2 `domain_aware`

- reference：同领域但不同措辞的隐私 / 模型 / 推理文本
- private：隐私 / 模型 / 推理主题文本

### 4.3 `distribution_aware`

- reference：与 private 同分布、不同样本的文本
- private：同分布另一份样本

这样可以保证：

- 三种知识设置严格递进；
- 不依赖外部下载；
- 每次实验都可完全复现。

---

## 5. 工作包

## Work Package 4.1：本地可复现语料生成

任务：

- 实现三种知识设置对应的语料生成器
- 输出 token 序列级语料

验收：

- 三种设置都能稳定生成 reference / private corpus

## Work Package 4.2：TFMA 最小实现

任务：

- 实现 unigram frequency matching
- 导出统一结果

验收：

- `run_tfma.py` 能真实执行

## Work Package 4.3：SDA 最小实现

任务：

- 实现 unigram + bigram signature matching
- 实现 token-level BLEU-4

验收：

- `run_sda.py` 能真实执行

## Work Package 4.4：批量比较

任务：

- 对所有 target 跑三种知识设置
- 导出比较结果

验收：

- 有 `tfma_comparison.json`
- 有 `sda_comparison.json`

## Work Package 4.5：文档收口

任务：

- 写 Gate 4 结果记录
- 更新推进看板

验收：

- Gate 4 结论可独立阅读

---

## 6. Gate 4 验收标准

Gate 4 完成必须满足：

- [ ] `run_tfma.py` 可真实执行
- [ ] `run_sda.py` 可真实执行
- [ ] 三种知识设置均有真实结果
- [ ] 至少 4 个 target 有真实结果
- [ ] 输出符合统一 schema
- [ ] 有 baseline vs 混淆对比
- [ ] 有 Gate 4 文档沉淀

---

## 7. 一句话结论

> Gate 4 的闭环重点是回答：即使不碰权重、不碰 embedding、不碰中间态，攻击者只凭长期在线观察到的 obfuscated token 统计信息，究竟能恢复多少 plaintext 结构。 
