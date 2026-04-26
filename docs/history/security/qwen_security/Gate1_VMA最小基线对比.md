# Gate 1：VMA 基线对比记录

## 1. 目标

本阶段的目标是把 `VMA` 做成一个**真实可运行、可比较、能逐步增强**的攻击基线，并重点回答：

1. `Stage A` 作为弱保护 baseline 时，恢复率有多高；
2. `Stage H` 研究线与 `Stage J / K` 部署线相比，谁更容易被恢复；
3. 当攻击从 `direct-only` 升级为 `projection-enhanced` 后，风险排序是否变化。

---

## 2. 当前已实现的两档 VMA

### 2.1 Direct-only 基线

只使用：

- `embed`
- `head`

方法：

- 对 token 相关权重行做排序；
- 提取分位点签名；
- 做 row-wise 标准化后匹配；
- 对来源分数做简单投票。

对应结果文件：

- `outputs/security_qwen/summary/vma_comparison_direct_only.json`

### 2.2 Projection-enhanced 基线

在 `direct-only` 的基础上，再加入：

- 多层 `q / k / v / gate / up` 投影源
- 多层分数投票

对应结果文件：

- `outputs/security_qwen/summary/vma_comparison_projection_enhanced.json`

这档攻击更接近论文中“利用多种权重关系恢复 permutation”的方向，因此应视为当前更强的 Gate 1 基线。

---

## 3. 对比对象

本轮统一比较：

- `stage_a_standard`
- `stage_h_full_obfuscated`
- `stage_j_stable_reference`
- `stage_j_tiny_a`
- `stage_k_stable_reference`
- `stage_k_tiny_a`

---

## 4. Direct-only 结果

| target | stage | profile | top1 | top10 |
|---|---|---:|---:|---:|
| `stage_a_standard` | A | - | `0.9765625` | `0.9921875` |
| `stage_j_stable_reference` | J | `stable_reference` | `0.1484375` | `0.3203125` |
| `stage_k_stable_reference` | K | `stable_reference` | `0.1484375` | `0.3203125` |
| `stage_j_tiny_a` | J | `tiny_a` | `0.140625` | `0.3203125` |
| `stage_k_tiny_a` | K | `tiny_a` | `0.140625` | `0.3203125` |
| `stage_h_full_obfuscated` | H | - | `0.0078125` | `0.0078125` |

### 4.1 这一档的解读

- `Stage A` 几乎可被直接恢复，符合弱保护 baseline 的定位；
- `Stage H` 几乎压到 0，说明研究线在 direct-only 攻击下很稳；
- `Stage J / K` 位于中间，说明部署线安全性弱于 `Stage H`，但明显强于 `Stage A`。

在只看这一档时，一个很自然的结论会是：

> 部署线虽然弱于研究线，但仍然显著优于 token-only baseline。

---

## 5. Projection-enhanced 结果

| target | stage | profile | top1 | top10 |
|---|---|---:|---:|---:|
| `stage_a_standard` | A | - | `0.984375` | `1.0` |
| `stage_j_stable_reference` | J | `stable_reference` | `0.9765625` | `0.984375` |
| `stage_j_tiny_a` | J | `tiny_a` | `0.9765625` | `0.984375` |
| `stage_k_stable_reference` | K | `stable_reference` | `0.9765625` | `0.984375` |
| `stage_k_tiny_a` | K | `tiny_a` | `0.9765625` | `0.984375` |
| `stage_h_full_obfuscated` | H | - | `0.1015625` | `0.2421875` |

### 5.1 这一档的解读

这轮结果比 direct-only 明显更重要。

它说明：

- `Stage A` 仍然几乎完全可恢复；
- `Stage H` 虽然恢复率上升，但仍远低于 `Stage A`；
- **`Stage J / K` 一旦纳入多层投影源，恢复率几乎追到 `Stage A`。**

这意味着：

> 如果攻击者不只盯着 embed/head，而是联合利用 attention / FFN 相关权重关系，那么当前 `Stage J / K` 部署线在 `VMA` 下的脆弱性会显著上升。

---

## 6. Baseline vs 混淆：当前最关键的比较

### 6.1 `Stage A` vs `Stage H`

- 在 direct-only 和 projection-enhanced 两档下，`Stage H` 都明显优于 `Stage A`；
- 说明研究线仍保留了更强的抗 `VMA` 能力。

### 6.2 `Stage A` vs `Stage J / K`

- 在 direct-only 下，`Stage J / K` 显著优于 `Stage A`；
- 但在 projection-enhanced 下，`Stage J / K` 与 `Stage A` 已经非常接近。

这条对比非常关键，因为它说明：

> 对 `Stage J / K` 而言，只看 direct-only 基线会低估风险；  
> 更强一点的 `VMA` 组合后，部署线可能已经接近 token-only baseline 的风险水平。

### 6.3 `Stage J` vs `Stage K`

当前两者几乎一致。

这表明：

- `Stage K` 包装本身没有额外显著放大 `VMA` 风险；
- `Stage K` 主要继承了底层 `Stage J` profile 的安全属性。

---

## 7. 当前阶段结论

当前 Gate 1 最重要的阶段性结论不是“部署线介于研究线和 baseline 之间”，而是更尖锐的一句：

> **在更强的 projection-enhanced VMA 下，`Stage J / K` 的 permutation 恢复风险远高于 direct-only 结论所显示的水平，并且已经逼近 `Stage A` baseline；而 `Stage H` 仍明显更稳。**

这意味着当前项目里最值得警惕的并不是：

- `Stage K` 包装层本身；

而是：

- **从研究线切换到 standard-shape / release 部署线之后，token-coupled 权重关系是否重新暴露了更强的可恢复结构。**

---

## 8. 当前边界

虽然这轮结论已经比最初那版更扎实，但仍不是最终版本，原因包括：

1. 当前仍是仓库内实现的 Gate 1 基线，不是论文完整 VMA；
2. 当前还没有逐个分析“哪类投影源贡献最大”；
3. 当前还没有做单层 / 多层投票消融；
4. 当前还没有结合 IA / IMA / ISA 交叉验证。

因此当前更准确的说法是：

> Gate 1 已经成功找到一个足以明显放大部署线风险的 VMA 基线，但还需要继续做 source attribution 和多层消融，才能把结论坐实到更细粒度层面。

---

## 9. 下一步

Gate 1 后续建议优先做：

1. **source attribution**
   - 分析 `q / k / v / gate / up` 哪类源贡献最大
2. **layer ablation**
   - 单层 vs 多层
   - 早层 vs 中层 vs 末层
3. **profile sensitivity**
   - `stable_reference` vs `tiny_a` 是否在更大样本下仍无显著差异

完成这些后，再进入 `Gate 2 / IMA` 会更稳。

---

## 10. 一句话结论

> 当前 Gate 1 已经把 `baseline vs 混淆` 的对比推进到一个更可信的层次：`Stage H` 研究线仍明显更稳，而 `Stage J / K` 在更强的 projection-enhanced VMA 下已接近 `Stage A` 弱保护 baseline，这提示部署线存在需要重点跟进的结构恢复风险。 
