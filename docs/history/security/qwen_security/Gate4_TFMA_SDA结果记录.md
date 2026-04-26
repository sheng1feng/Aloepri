# Gate 4：TFMA + SDA 结果记录

## 1. 目标

Gate 4 的目标是回答：

1. 长期在线观察 obfuscated token 频率，攻击者能恢复多少 plaintext token；
2. 引入序列共现结构后，攻击是否会显著增强；
3. `Stage A/H/J/K` 在频率攻击下是否存在差异。

---

## 2. 当前实现

### 2.1 TFMA

最小 `TFMA`：

- 统计 reference plaintext corpus 的 unigram 频率；
- 统计 private obfuscated corpus 的 unigram 频率；
- 基于频率排序和频率距离做 matching；
- 输出：
  - `token_top1_recovery_rate`
  - `token_top10_recovery_rate`
  - `token_top100_recovery_rate`
  - `sensitive_token_recovery_rate`

### 2.2 SDA

最小 `SDA`：

- 在 unigram 的基础上加入 sorted outgoing / incoming bigram signature；
- 进行 substitution-style matching；
- 用 top1 映射重建 private corpus token 序列；
- 输出：
  - `token_top1_recovery_rate`
  - `token_top10_recovery_rate`
  - `token_top100_recovery_rate`
  - `bleu4`

### 2.3 三种知识设置

- `zero_knowledge`
- `domain_aware`
- `distribution_aware`

当前使用本地可复现语料生成器，不依赖外部下载。

---

## 3. TFMA 结果

结果文件：

- `outputs/security_qwen/summary/tfma_comparison.json`

### 3.1 Zero-knowledge

所有 target 一致：

- `top1 = 0.0689655`
- `top10 = 0.1034483`
- `top100 = 1.0`

### 3.2 Domain-aware

所有 target 一致：

- `top1 = 0.2413793`
- `top10 = 0.7241379`
- `top100 = 1.0`

### 3.3 Distribution-aware

所有 target 一致：

- `top1 = 0.2413793`
- `top10 = 0.9310345`
- `top100 = 1.0`

### 3.4 TFMA 当前解读

这轮最重要的结论不是具体数值本身，而是：

> **`Stage A / H / J / K` 在 TFMA 下几乎完全一致。**

这说明当前最小 `TFMA` 所利用的信息主要是：

- token permutation 诱导出的在线 token 频率结构

而不是：

- hidden transform
- KeyMat
- standard-shape
- release 包装

换句话说：

> 对频率攻击来说，后续 hidden-space 混淆基本不改变攻击面；  
> 频率风险几乎完全由 token-level obfuscation 决定。

同时三种知识设置的差异也很清楚：

- `zero_knowledge` 最弱
- `domain_aware` 明显增强
- `distribution_aware` 进一步增强

---

## 4. SDA 结果

结果文件：

- `outputs/security_qwen/summary/sda_comparison.json`

### 4.1 Zero-knowledge

所有 target 一致：

- `top1 = 0.0344828`
- `top10 = 0.0689655`
- `top100 = 1.0`
- `bleu4 ≈ 5.87e-07`

### 4.2 Domain-aware

所有 target 一致：

- `top1 = 0.0344828`
- `top10 = 0.2413793`
- `top100 = 1.0`
- `bleu4 ≈ 5.87e-07`

### 4.3 Distribution-aware

所有 target 一致：

- `top1 = 0.3103448`
- `top10 = 1.0`
- `top100 = 1.0`
- `bleu4 ≈ 0.09193`

### 4.4 SDA 当前解读

这轮 `SDA` 告诉我们两点：

1. 一旦 reference 与 private 分布足够接近，序列结构攻击会明显增强；
2. 即便 token Top-k 恢复率变高，当前 token-level `BLEU-4` 仍不算高。

更重要的是：

> **和 TFMA 一样，`SDA` 在 `Stage A/H/J/K` 上也几乎完全一致。**

这再次说明：

- 频率/共现攻击的主导因素是 token permutation；
- 后续 hidden-space 路线对这类攻击几乎没有额外缓解。

---

## 5. 与前 3 个 Gate 的关系

把四条攻击线合起来看：

### Gate 1 / VMA

- 暴露结构恢复风险
- `Stage J / K` 明显弱于 `Stage H`

### Gate 2 / IMA

- 暴露 embedding 表示恢复风险
- 所有 target 都高度可恢复

### Gate 3 / ISA

- 当前 hidden-state ISA 有限有效
- attention-score 基线几乎无效

### Gate 4 / TFMA + SDA

- 风险主要由 token permutation 决定
- `Stage A/H/J/K` 几乎无差异

因此当前一个非常关键的综合结论是：

> **不同阶段工件之间真正拉开差距的，是结构恢复类攻击；而频率攻击几乎只由 token-level obfuscation 决定。**

---

## 6. 当前结论

### 6.1 频率攻击不是最危险，但也不是可以忽略

- `zero_knowledge` 下，TFMA / SDA 都较弱；
- `domain_aware` / `distribution_aware` 下明显变强；
- 说明频率泄露在有先验时是真实风险。

### 6.2 频率攻击不区分研究线与部署线

这是 Gate 4 最重要的项目结论：

> **`Stage H` 不会像在 VMA 里那样天然优于 `Stage J / K`。**

因为这条攻击线几乎完全发生在：

- token obfuscation
- token 序列统计

而不发生在：

- hidden-state transform
- FFN / attention 混淆

### 6.3 当前风险排序

基于目前 4 个 Gate 的结果，一个较稳妥的排序是：

1. `IMA`：最强、最广泛
2. `VMA`：对部署线尤其危险
3. `TFMA / SDA`：对所有线一致，受先验影响显著
4. 当前最小 `ISA`：相对较弱

---

## 7. Gate 4 完成判定

对照 `Gate4_TFMA_SDA闭环计划.md`，当前已经满足：

- [x] `run_tfma.py` 可真实执行
- [x] `run_sda.py` 可真实执行
- [x] 三种知识设置均有真实结果
- [x] 至少 4 个 target 有真实结果
- [x] 输出符合统一 schema
- [x] 有 baseline vs 混淆对比
- [x] 有 Gate 4 文档沉淀

因此当前可以正式给出结论：

> **Gate 4 已完成。**

---

## 8. 下一步

完成 Gate 4 后，下一步建议进入：

- `Gate 5 / 参数-安全性联合扫描`

因为现在最合理的问题已经变成：

- `alpha_e / alpha_h / lambda` 对 `VMA / IMA / ISA / TFMA/SDA` 分别有什么影响；
- 是否存在真正可接受的 accuracy-privacy tradeoff；
- 当前推荐工作点到底应该落在哪。

---

## 9. 一句话结论

> 当前 Gate 4 已完成，并明确表明：频率类攻击的风险几乎完全由 token permutation 决定，因此 `Stage A/H/J/K` 在这条线上几乎无差异；这说明后续 hidden-space 混淆并不能自然缓解长期在线统计泄露。 
