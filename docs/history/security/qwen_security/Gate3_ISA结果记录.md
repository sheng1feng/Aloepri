# Gate 3：ISA 结果记录

## 1. 目标

Gate 3 的目标是完成一条**部署态中间量攻击**闭环，用来回答：

1. 当服务端在推理过程中可见部分中间量时，是否能恢复 plaintext token；
2. `Stage H`、`Stage J / K` 在中间态攻击下的风险排序是否与 `Gate 1 / Gate 2` 一致；
3. 对当前系统来说，中间态攻击是否比 embedding 级反演更危险。

---

## 2. 当前实现

本轮 ISA 使用的是一个**最小 ridge 反演器**：

- 输入：部署态可见 observable
  - `hidden_state`
  - `attention_score`
- 标签：plaintext embedding
- 恢复方式：
  1. `observable -> plaintext embedding`
  2. 最近邻检索恢复 plaintext token id

当前统一比较：

- `stage_a_standard`
- `stage_h_full_obfuscated`
- `stage_j_stable_reference`
- `stage_j_tiny_a`
- `stage_k_stable_reference`
- `stage_k_tiny_a`

---

## 3. Hidden-State 结果

结果文件：

- `outputs/security_qwen/summary/isa_hidden_state_comparison.json`

当前结果：

| target | stage | profile | top1 | top10 | cosine | vs Stage A top1 |
|---|---|---:|---:|---:|---:|---:|
| `stage_a_standard` | A | - | `0.0390625` | `0.0625` | `0.1363861` | `0.0` |
| `stage_j_stable_reference` | J | `stable_reference` | `0.0390625` | `0.0703125` | `0.1364365` | `0.0` |
| `stage_j_tiny_a` | J | `tiny_a` | `0.0390625` | `0.0546875` | `0.1384937` | `0.0` |
| `stage_k_stable_reference` | K | `stable_reference` | `0.0390625` | `0.0703125` | `0.1364365` | `0.0` |
| `stage_k_tiny_a` | K | `tiny_a` | `0.0390625` | `0.0546875` | `0.1384937` | `0.0` |
| `stage_h_full_obfuscated` | H | - | `0.0078125` | `0.046875` | `0.1409482` | `-0.03125` |

### 3.1 解读

- 当前最小 hidden-state ISA 的恢复率整体不高；
- `Stage H` 仍然优于 `Stage A / J / K`；
- `Stage J / K` 与 `Stage A` 几乎没有可分辨差异。

这说明：

> 在当前最小 hidden-state 攻击设置下，部署线没有像 Gate 1 那样出现“显著高于 Stage A”的额外恶化；
> 但研究线 `Stage H` 依然表现出更低的 token 恢复率。

---

## 4. Attention-Score 结果

结果文件：

- `outputs/security_qwen/summary/isa_attention_score_comparison.json`

当前结果：

| target | stage | profile | top1 | top10 | cosine | vs Stage A top1 |
|---|---|---:|---:|---:|---:|---:|
| `stage_a_standard` | A | - | `0.0` | `0.0` | `0.2083425` | `0.0` |
| `stage_h_full_obfuscated` | H | - | `0.0` | `0.0` | `0.2078137` | `0.0` |
| `stage_j_stable_reference` | J | `stable_reference` | `0.0` | `0.0` | `0.2083416` | `0.0` |
| `stage_j_tiny_a` | J | `tiny_a` | `0.0` | `0.0` | `0.2082648` | `0.0` |
| `stage_k_stable_reference` | K | `stable_reference` | `0.0` | `0.0` | `0.2083416` | `0.0` |
| `stage_k_tiny_a` | K | `tiny_a` | `0.0` | `0.0` | `0.2082648` | `0.0` |

### 4.1 解读

当前最小 attention-score ISA 基本没有恢复能力。

这意味着在当前设置下：

- 单纯依赖 attention score；
- 使用最小 ridge 反演器；
- 在当前数据规模与候选池下；

攻击者无法有效恢复 plaintext token。

更准确地说：

> 当前 attention-score 最小基线太弱，尚不足以构成有效攻击。

---

## 5. 与 Gate 1 / Gate 2 的关系

把前三个 Gate 放在一起看：

### Gate 1 / VMA

- 暴露了**结构恢复**风险
- `Stage J / K` 在更强 VMA 下明显更脆弱

### Gate 2 / IMA

- 暴露了**embedding 语义可学习恢复**风险
- 所有 target 都表现出很高恢复率

### Gate 3 / ISA

- 暴露了**部署态中间量**风险
- 当前 hidden-state 攻击有效但不强
- 当前 attention-score 攻击几乎无效

因此当前最重要的综合判断是：

> 目前最危险的不是当前最小 ISA，而是已经在 Gate 1 / Gate 2 中观察到的结构恢复与 embedding 语义恢复。

---

## 6. 当前结论

### 6.1 关于 hidden_state

- `Stage H` 优于 `Stage A / J / K`
- `Stage J / K` 与 `Stage A` 接近
- 说明研究线在 hidden-state 保护上仍有一定优势

### 6.2 关于 attention_score

- 当前最小攻击没有打出有效恢复率
- 暂时不能说明 attention_score 一定安全
- 只能说明：
  - 当前最小 ridge 基线不够强
  - 当前设置下 attention score 不像 `VMA / IMA` 那样危险

### 6.3 Gate 3 的总体判断

> 与 `VMA / IMA` 相比，当前最小 ISA 并没有成为最危险攻击面。  
> 当前系统更紧迫的风险仍然是：
> 1. `VMA` 暴露的结构恢复  
> 2. `IMA` 暴露的 embedding 表示恢复

---

## 7. Gate 3 完成判定

对照 `Gate3_ISA闭环计划.md`，当前已经满足：

- [x] `run_isa.py` 可真实执行
- [x] `hidden_state` 有真实结果
- [x] `attention_score` 有真实结果
- [x] 至少 4 个 target 有真实结果
- [x] 输出符合统一 schema
- [x] 有 baseline vs 混淆对比
- [x] 有 Gate 3 文档沉淀

因此当前可以正式给出结论：

> **Gate 3 已完成。**

---

## 8. 下一步

完成 Gate 3 后，下一步建议进入：

- `Gate 4 / TFMA + SDA`

因为现在最合理的问题已经变成：

- 长期在线统计攻击在当前系统中的风险级别如何；
- 它是否会与 `VMA / IMA` 一样形成高等级风险；
- 四条攻击线里，哪条才是当前真正优先级最高的安全短板。

---

## 9. 一句话结论

> 当前 Gate 3 已完成，并给出一个清晰信号：在最小部署态 observable 攻击设置下，`hidden_state` 存在有限恢复能力且 `Stage H` 仍更稳，而 `attention_score` 基线几乎无效；因此当前系统更紧迫的风险仍来自 Gate 1 的结构恢复和 Gate 2 的 embedding 语义恢复。 
