# Gate 2：IMA 结果记录

## 1. 目标

Gate 2 的目标是完成一条**训练式反演攻击**闭环，回答：

1. 当前各类工件的 obfuscated embedding 是否仍然能被学习性地映射回 plaintext 语义空间；
2. `Stage A`、`Stage H`、`Stage J / K` 在训练式反演下的风险排序如何；
3. `VMA` 的结构恢复结论与 `IMA` 的学习式反演结论是否一致。

---

## 2. 当前实现

本轮实现的是一个**最小但真实可运行**的 `IMA`：

- 输入：obfuscated embedding row
- 标签：plaintext embedding row
- 模型：ridge inversion model
- 推理方式：
  1. 从 obfuscated embedding 回归 plaintext embedding
  2. 在候选 plaintext embedding 池中做最近邻检索
  3. 输出 token Top-k 恢复率与 embedding cosine similarity

这是一种“线性训练式反演器”。

它不等同于论文里更重的神经反演模型，但已经足够回答：

- obfuscated embedding 是否还保留强可学习结构；
- 哪类工件在表示空间上更容易被学回来。

---

## 3. 对比对象

本轮统一比较：

- `stage_a_standard`
- `stage_h_full_obfuscated`
- `stage_j_stable_reference`
- `stage_j_tiny_a`
- `stage_k_stable_reference`
- `stage_k_tiny_a`

结果文件：

- `outputs/security_qwen/summary/ima_comparison.json`
- `outputs/security_qwen/ima/stage_a_standard.json`
- `outputs/security_qwen/ima/stage_h_full_obfuscated.json`
- `outputs/security_qwen/ima/stage_j_stable_reference.json`
- `outputs/security_qwen/ima/stage_j_tiny_a.json`
- `outputs/security_qwen/ima/stage_k_stable_reference.json`
- `outputs/security_qwen/ima/stage_k_tiny_a.json`

---

## 4. 当前结果

| target | stage | profile | top1 | top10 | cosine | vs Stage A top1 |
|---|---|---:|---:|---:|---:|---:|
| `stage_a_standard` | A | - | `0.9765625` | `1.0` | `0.9996496` | `0.0` |
| `stage_h_full_obfuscated` | H | - | `0.9765625` | `0.9765625` | `0.9507417` | `0.0` |
| `stage_j_stable_reference` | J | `stable_reference` | `0.9765625` | `0.9921875` | `0.9996496` | `0.0` |
| `stage_j_tiny_a` | J | `tiny_a` | `0.9765625` | `0.9765625` | `0.9978541` | `0.0` |
| `stage_k_stable_reference` | K | `stable_reference` | `0.9765625` | `0.9921875` | `0.9996496` | `0.0` |
| `stage_k_tiny_a` | K | `tiny_a` | `0.9765625` | `0.9765625` | `0.9978541` | `0.0` |

补充字段：

- `sensitive_token_recovery_rate` 当前均为 `1.0`
- `selected_ridge_alpha`：
  - `Stage A / J stable / K stable` 为 `1.0`
  - `Stage H / J tiny / K tiny` 为 `0.0001`

---

## 5. 当前解读

## 5.1 Gate 2 的主结论与 Gate 1 很不同

Gate 1 的结论是：

- `Stage H` 明显强于 `Stage J / K`
- `Stage J / K` 在更强 `VMA` 下接近 `Stage A`

但 Gate 2 的当前最小 `IMA` 给出的信号是：

> **在 token-row 级最小训练式反演设置下，几乎所有工件都高度可恢复。**

也就是说：

- 对 `IMA` 而言，`Stage H` 没有像在 `VMA` 里那样显著领先；
- `Stage J / K` 也没有进一步恶化到显著高于 `Stage H`；
- 当前线性反演器几乎对所有 target 都有效。

## 5.2 这说明什么

这说明当前 Gate 2 更像在测：

- obfuscated embedding 是否仍保留一个**稳定可学习映射**

而不是在测：

- token permutation 是否容易被直接恢复。

换句话说：

> Gate 1 主要暴露“结构恢复风险”；  
> Gate 2 主要暴露“表示空间仍然高度可学习”的风险。

## 5.3 `Stage H` 的细节

虽然 `Stage H` 的 `top1` 与其他 target 一样高，但它的：

- `embedding_cosine_similarity = 0.9507`

明显低于：

- `Stage A / J stable / K stable ≈ 0.9996`
- `J/K tiny ≈ 0.9979`

这说明：

- `Stage H` 并不是“和其他完全一样”；
- 只是当前 `candidate_pool_size=2048` 下，线性反演器已经足以把 token id 找对。

因此当前更准确的判断是：

> `Stage H` 在表示空间上仍有一定扰动效果，但不足以阻止当前最小 IMA 将 token 恢复到极高水平。

## 5.4 profile 差异

当前 `stable_reference` 与 `tiny_a` 在 `IMA` 下也没有拉开明显差距。

这说明至少在当前最小线性反演器设置下：

- profile 噪声并没有显著改变恢复率层级；
- 更细差异需要更强反演器或更严格检索设置再观察。

---

## 6. 当前边界

Gate 2 当前仍有以下边界：

1. 当前是 token-row proxy 设置，不是完整公开文本语料反演；
2. 当前模型是线性 ridge 反演器，不是更强的神经网络反演器；
3. 当前候选池大小有限；
4. 当前没有做 train-size / candidate-pool-size 敏感性扫描。

因此当前更准确的说法是：

> Gate 2 已经成功证明“最小训练式反演器足以高度恢复当前工件的 embedding 语义”，但还没有把这种风险刻画到更强模型、更大候选池和更真实公开语料设置下。

---

## 7. Gate 2 完成判定

对照 `Gate2_IMA闭环计划.md`，当前已经满足：

- [x] `run_ima.py` 可真实执行
- [x] 至少 4 个 target 有真实结果
- [x] 输出符合统一 schema
- [x] 有 `token_top1_recovery_rate`
- [x] 有 `token_top10_recovery_rate`
- [x] 有 `embedding_cosine_similarity`
- [x] 有 baseline vs 混淆对比
- [x] 有 Gate 2 文档沉淀

因此当前可以正式给出结论：

> **Gate 2 已完成。**

---

## 8. 与 Gate 1 合并后的项目判断

把 Gate 1 与 Gate 2 合起来看，目前可以得到一个更完整的判断：

- `Gate 1 / VMA`：
  - 研究线与部署线有明显差异
  - 部署线在结构恢复上更脆弱

- `Gate 2 / IMA`：
  - 训练式反演对所有线都很强
  - 表示空间的可学习泄露比预期更严重

因此当前项目最重要的中间结论是：

> **`Stage H` 相对 `Stage J / K` 的优势主要体现在结构恢复抗性上；但在最小训练式反演设置下，各条线都表现出很强的 embedding 语义可恢复性。**

---

## 9. 下一步

完成 Gate 2 后，下一步建议进入：

- `Gate 3 / ISA`

重点应回答：

1. 部署态可见中间量是否比 embedding 本身更危险；
2. `Stage H` 的中间态保护是否仍然优于 `Stage J / K`；
3. 中间态攻击与 `VMA / IMA` 的风险主次关系如何。

---

## 10. 一句话结论

> 当前 Gate 2 已完成，并给出一个重要警告：即使不看结构恢复，仅用最小线性训练式反演器，也足以高度恢复 `Stage A / H / J / K` 的 token-level embedding 语义，这说明当前系统的表示空间泄露风险非常高。 
