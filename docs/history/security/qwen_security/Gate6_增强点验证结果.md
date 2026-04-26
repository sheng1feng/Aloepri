# Gate 6：增强点验证结果

## 1. 目标

Gate 6 的目标是验证：

> 是否存在一种比“全局继续调大噪声”更有针对性的增强点，能在不明显破坏默认正确性的情况下，实质降低 `VMA / IMA` 风险。

本轮实现的增强点是：

- **Sensitive-Token Targeted Hardening**

也就是：

- 只对敏感 token 子集施加更强 embedding/head noise；
- 非敏感 token 维持原有工作点。

---

## 2. 验证对象

比较对象：

- `stable_reference`
- `tiny_a`
- `targeted_mild`
- `targeted_strong`
- `targeted_extreme`

结果文件：

- `outputs/security_qwen/summary/gate6_enhancement_scan.json`

---

## 3. 当前结果

| case | gid_exact | logits_max | vma_top1 | vma_sensitive_top1 | ima_top1 | ima_sensitive_top1 |
|---|---:|---:|---:|---:|---:|---:|
| `stable_reference` | `1.0` | `0.000126` | `0.96875` | `1.0` | `0.9765625` | `1.0` |
| `tiny_a` | `1.0` | `0.671531` | `0.96875` | `1.0` | `0.96875` | `1.0` |
| `targeted_mild` | `1.0` | `1.338077` | `0.96875` | `1.0` | `0.9765625` | `1.0` |
| `targeted_strong` | `1.0` | `2.676179` | `0.96875` | `1.0` | `0.9765625` | `1.0` |
| `targeted_extreme` | `0.8` | `26.761971` | `0.84375` | `0.0` | `0.96875` | `0.9375` |

---

## 4. 当前解读

## 4.1 `targeted_mild / targeted_strong`

当前结果显示：

- correctness 保持 `1.0`
- 但 `VMA / IMA` 完全没有改善
- `sensitive_token_recovery_rate` 也没有下降

这说明：

> 在可接受的轻量定向加噪区间内，当前增强点**没有实质收益**。

## 4.2 `targeted_extreme`

当前结果显示：

- `VMA sensitive_top1` 从 `1.0` 降到 `0.0`
- `VMA overall_top1` 从 `0.96875` 降到 `0.84375`
- 但：
  - `generated_ids_exact_match_rate` 降到 `0.8`
  - `IMA sensitive_top1` 只从 `1.0` 降到 `0.9375`

这说明：

> 极端定向加噪确实能明显压低敏感 token 的 `VMA` 恢复率；  
> 但它带来的正确性损失已经不可忽略，而且对 `IMA` 的改善非常有限。

---

## 5. Gate 6 的核心结论

当前最重要的不是“Gate 6 找到一个好方案”，而是：

> **Gate 6 证明了当前这类“只对敏感 token 行加更强噪声”的增强点，并不能在可接受正确性区间内有效同时压低 `VMA` 和 `IMA`。**

更具体地说：

1. 轻量定向加噪：几乎无收益
2. 极端定向加噪：能压 `VMA`，但正确性受损，且 `IMA` 仍很强

因此当前更稳妥的结论是：

> 这条增强路线可以作为“特定高敏 token 的应急加固手段”研究下去，  
> 但它**还不足以成为当前系统的推荐增强方案**。

---

## 6. 项目级结论（截至 Gate 6）

把 6 个 Gate 合起来看：

- `Gate 1 / VMA`
  - 部署线结构恢复风险高
- `Gate 2 / IMA`
  - 表示恢复风险高，且对所有线都很强
- `Gate 3 / ISA`
  - 当前最小中间态攻击相对较弱
- `Gate 4 / TFMA + SDA`
  - 风险由 token permutation 主导
- `Gate 5 / 参数扫描`
  - 可接受参数区间内很难换来明显安全收益
- `Gate 6 / 增强点验证`
  - 简单的敏感 token 定向加噪不足以解决核心问题

所以当前最重要的项目判断已经相当明确：

> **现有方法路径的主要瓶颈不是“参数没调好”，也不是“还缺一个简单补丁”，而是结构恢复与表示恢复本身都存在很强攻击面。**

---

## 7. Gate 6 完成判定

对照 `Gate6_增强点验证计划.md`，当前已经满足：

- [x] 至少一个增强 artifact 成功导出
- [x] 有默认 prompt 正确性结果
- [x] 有 `VMA` 对比结果
- [x] 有 `IMA` 对比结果
- [x] 有 `sensitive_token_recovery_rate` 对比
- [x] 有 Gate 6 文档沉淀

因此当前可以正式给出结论：

> **Gate 6 已完成。**

---

## 8. 下一步建议

当前再继续“微调已有路径”收益可能已经很低。

更值得考虑的下一步是：

1. 总结总报告
2. 提炼当前最关键的开放问题
3. 如果继续做方法增强，应考虑更结构性的改动，而不是再做轻量补丁

---

## 9. 一句话结论

> 当前 Gate 6 已完成，并明确表明：简单的敏感 token 定向加噪不能在可接受正确性区间内同时显著压低 `VMA` 和 `IMA`，因此它不足以作为当前系统的有效增强方案。 
