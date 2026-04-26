# Gate 5：参数-安全性联合扫描结果

## 1. 目标

Gate 5 的目标是把前面各类攻击结果与 `Stage J` 的参数工作点统一到一张表里，回答：

1. `alpha_e / alpha_h` 变化时，accuracy 与 security 如何联动；
2. 当前最合理的非零工作点是什么；
3. 哪些攻击随参数变化而变，哪些几乎不变。

---

## 2. 扫描对象

本轮扫描点：

- `stable_reference`
- `tiny_a`
- `tiny_b`
- `small_a`
- `paper_like`

对应参数：

| case | alpha_e | alpha_h |
|---|---:|---:|
| `stable_reference` | `0.0` | `0.0` |
| `tiny_a` | `0.02` | `0.01` |
| `tiny_b` | `0.05` | `0.02` |
| `small_a` | `0.1` | `0.05` |
| `paper_like` | `1.0` | `0.2` |

结果文件：

- `outputs/security_qwen/summary/gate5_parameter_scan.json`

---

## 3. 当前结果

| case | gid_exact | logits_max | vma_top1 | ima_top1 | isa_hidden_top1 |
|---|---:|---:|---:|---:|---:|
| `stable_reference` | `1.0` | `0.000126` | `0.9765625` | `0.9765625` | `0.0390625` |
| `tiny_a` | `1.0` | `0.671531` | `0.9765625` | `0.9765625` | `0.0390625` |
| `tiny_b` | `1.0` | `1.687374` | `0.9765625` | `0.9765625` | `0.046875` |
| `small_a` | `0.6` | `3.441067` | `0.9765625` | `0.9765625` | `0.03125` |
| `paper_like` | `0.0` | `17.032056` | `0.109375` | `0.90625` | `0.0` |

补充说明：

- `tfma_domain_top10` 当前在所有点上基本不变；
- `sda_distribution_bleu4` 当前在所有点上基本不变；
- 这符合 Gate 4 的结论：频率攻击主要由 token permutation 决定，而不是由 `alpha_e / alpha_h` 决定。

---

## 4. Accuracy-Privacy Tradeoff 解读

## 4.1 从 `stable_reference` 到 `tiny_a / tiny_b`

观察：

- correctness 基本保持：
  - `generated_ids_exact_match_rate = 1.0`
- 但 `VMA / IMA` 几乎没有下降：
  - `vma_top1 ≈ 0.9765625`
  - `ima_top1 ≈ 0.9765625`

这说明：

> 在当前 Gate 5 扫描下，小幅增加 `alpha_e / alpha_h` 并没有带来可见的安全收益。  
> 也就是说，`tiny_a / tiny_b` 相比 `stable_reference` 的主要变化体现在数值误差，而不是攻击面实质下降。

## 4.2 `small_a`

观察：

- correctness 已明显下降：
  - `generated_ids_exact_match_rate = 0.6`
- 但 `VMA / IMA` 仍几乎不变：
  - `vma_top1 = 0.9765625`
  - `ima_top1 = 0.9765625`

这说明：

> 对当前部署线来说，把 `alpha_e / alpha_h` 推到 `small_a` 这一级别，先损失的是功能正确性，而不是显著换来安全性提升。

## 4.3 `paper_like`

观察：

- correctness 基本崩溃：
  - `generated_ids_exact_match_rate = 0.0`
- `VMA` 显著下降：
  - `vma_top1 = 0.109375`
- `IMA` 虽然下降，但仍然很高：
  - `ima_top1 = 0.90625`

这说明：

> 真正把参数推到论文级强噪声点时，确实能明显压低结构恢复类攻击；  
> 但训练式 embedding 反演仍然很强，而且正确性已经不可接受。

---

## 5. 当前最关键结论

当前 Gate 5 最重要的结论不是“更大噪声更安全”这么简单，而是更具体的三点：

### 5.1 在可用工作点内，安全收益非常有限

从 `stable_reference -> tiny_a -> tiny_b`：

- correctness 基本保持；
- `VMA / IMA / ISA(hidden)` 没有出现明显改善。

这说明在当前扫描区间内：

> **可接受的非零噪声工作点，并没有显著改变当前主要攻击风险。**

### 5.2 真正有明显安全变化的点，往往已经不可用

只有到 `paper_like`：

- `VMA` 才明显下降；
- 但 correctness 已不可接受。

这意味着当前系统面临一个典型 tradeoff：

> **想要显著压低结构恢复风险，需要付出过大的功能代价。**

### 5.3 频率攻击对参数点不敏感

当前 `TFMA / SDA` 基本不随参数变化。

这再次确认：

- Gate 4 的结论是稳的；
- 频率攻击属于 token permutation 层面的固有风险；
- `alpha_e / alpha_h` 调参几乎无法缓解它。

---

## 6. 推荐工作点

基于当前 Gate 5 扫描结果，当前推荐结论是：

### 6.1 correctness baseline

- `stable_reference`

适合：

- correctness 对照
- 安全基线分析
- 研究复现实验

### 6.2 推荐非零工作点

- `tiny_a`

原因：

- correctness 保持完整；
- 相比 `stable_reference` 至少保留了非零噪声；
- 与 `tiny_b / small_a` 相比，没有更差的安全收益但数值更稳。

更准确地说：

> `tiny_a` 不是因为它已经带来显著安全增益而被推荐；  
> 而是因为在当前可用非零工作点里，它是“**正确性最好、且不比更强噪声点更吃亏**”的折中点。

---

## 7. Gate 5 完成判定

对照 `Gate5_参数安全性联合扫描计划.md`，当前已经满足：

- [x] 至少 5 个参数点有统一结果
- [x] 同一张表上同时包含 accuracy 与 security 指标
- [x] 有推荐工作点结论
- [x] 有 Gate 5 文档沉淀

因此当前可以正式给出结论：

> **Gate 5 已完成。**

---

## 8. 综合判断（截至 Gate 5）

把前 5 个 Gate 放在一起看：

- `Gate 1 / VMA`
  - 结构恢复对部署线最危险
- `Gate 2 / IMA`
  - 训练式 embedding 反演对所有线都很强
- `Gate 3 / ISA`
  - 当前最小部署态中间量攻击相对较弱
- `Gate 4 / TFMA + SDA`
  - 频率攻击主要由 token permutation 决定
- `Gate 5 / 参数扫描`
  - 在可接受工作点区间内，调大 `alpha_e / alpha_h` 并未显著降低核心风险

因此当前最重要的项目级结论是：

> **当前系统的主要风险不是“工作点还没调对”，而是现有方法路径本身在结构恢复与表示恢复上都存在强攻击面。**

---

## 9. 下一步

完成 Gate 5 后，下一步建议进入：

- `Gate 6 / 增强点验证`

而且方向已经比较明确：

1. 不是继续盲目调 `alpha_e / alpha_h`
2. 而是要设计真正能压低：
   - `VMA`
   - `IMA`

的增强点

---

## 10. 一句话结论

> 当前 Gate 5 已完成，并清楚地表明：在当前可接受的正确性区间内，调大 `alpha_e / alpha_h` 几乎不能显著降低核心攻击风险；真正能明显降风险的强噪声点又会让正确性不可接受，因此当前问题更像是“方法路径的安全上限”，而不是“参数还没调对”。 
