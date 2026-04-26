# Gate 3：ISA 闭环计划

## 1. 目标

Gate 3 的目标是完成一条**部署态中间量攻击**闭环，用来回答：

1. 如果服务端在 client/server 分离部署中可见部分中间量，是否能比 `VMA / IMA` 更高效地恢复 plaintext token；
2. `Stage H`、`Stage J / K` 在中间态攻击下的风险排序是否与 `Gate 1 / Gate 2` 一致；
3. 对当前系统而言，更危险的是：
   - 静态结构恢复；
   - 训练式 embedding 反演；
   - 还是部署态可见中间量本身。

一句话理解：

> Gate 1 看结构，Gate 2 看表示，Gate 3 看“服务端在推理过程中看到的中间量”。

---

## 2. 本轮范围

### In scope

- `hidden_state`
- `attention_score`
- 统一比较：
  - `stage_a_standard`
  - `stage_h_full_obfuscated`
  - `stage_j_stable_reference`
  - `stage_j_tiny_a`
  - `stage_k_stable_reference`
  - `stage_k_tiny_a`

### Out of scope

- `kv_cache` 完整攻击
- 多轮对话上下文攻击
- 大规模生成式反演器
- 依赖本机开发 hook 文件的攻击假设

注意：

虽然本轮在本地实现中会用临时 instrumentation 抽取中间张量，但这只是为了**模拟服务端天然可见的部署态中间量**，不等于把“开发态 debug hook 暴露”算作攻击面。

---

## 3. 最小攻击方案

本轮 ISA 采用一个最小但可验证的闭环：

1. 生成一批 plaintext token 序列；
2. client 侧用 `perm_vocab` 将其映射为 obfuscated input ids；
3. 服务端模型在推理时暴露某个中间量：
   - `hidden_state`
   - `attention_score`
4. 攻击者训练一个最小 ridge 反演器：
   - `observable -> plaintext embedding`
5. 再通过最近邻检索恢复 plaintext token id。

这样做的优点是：

- 与 Gate 2 保持可比；
- 可以直接比较“embedding 本身”和“中间态”谁更危险；
- CPU 上也能快速闭环。

---

## 4. 工作包

## Work Package 3.1：统一 observable 抽取

任务：

- 为标准 HF 路线支持：
  - `output_hidden_states=True`
  - `output_attentions=True`
- 为 `Stage H` 支持：
  - 通过临时 forward instrumentation 提取部署态可见张量
- 统一输出 shape 到 token-row 级样本。

交付：

- `src/security_qwen/isa.py` 中的 observable 抽取函数

验收：

- `hidden_state` 与 `attention_score` 都能在至少一个 target 上成功抽取

## Work Package 3.2：最小反演器

任务：

- 复用 Gate 2 的最小 ridge 反演范式
- 支持从 observable 回归到 plaintext embedding
- 输出：
  - `intermediate_top1_recovery_rate`
  - `token_top10_recovery_rate`
  - `embedding_cosine_similarity`

交付：

- `run_isa_baseline(...)`

验收：

- 至少一种 observable 在至少一个 target 上能稳定得到真实结果

## Work Package 3.3：多 target 比较

任务：

- 统一跑 `Stage A/H/J/K`
- 分 observable 输出比较结果

交付：

- `scripts/security_qwen/export_isa_comparison.py`
- `outputs/security_qwen/summary/isa_hidden_state_comparison.json`
- `outputs/security_qwen/summary/isa_attention_score_comparison.json`

验收：

- 可直接回答不同 observable 和不同工件的风险排序

## Work Package 3.4：文档收口

任务：

- 写 Gate 3 结果记录
- 更新推进看板
- 说明 Gate 3 与 Gate 1 / Gate 2 的关系

验收：

- Gate 3 结论可独立阅读

---

## 5. Gate 3 验收标准

Gate 3 完成必须满足：

- [ ] `run_isa.py` 可真实执行
- [ ] `hidden_state` 有真实结果
- [ ] `attention_score` 有真实结果
- [ ] 至少 4 个 target 有真实结果
- [ ] 输出符合统一 schema
- [ ] 有 baseline vs 混淆对比
- [ ] 有 Gate 3 文档沉淀

---

## 6. 通过后下一步

若 Gate 3 完成，下一步进入：

- `Gate 4 / TFMA + SDA`

届时要回答：

- 长期在线统计攻击是否与中间态攻击同样危险；
- `Stage J / K` 在静态、训练式、中间态、频率攻击四条线上的综合风险排序是什么。

---

## 7. 一句话结论

> Gate 3 的闭环重点是用最小部署态 observable 攻击，回答“服务端在推理过程中天然能看到的中间量，是否比 embedding 本身更容易泄露 plaintext token”。 
