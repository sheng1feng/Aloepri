# Gate 6：增强点验证计划

## 1. 目标

Gate 6 的目标不是再扫参数，而是验证**真正可能改变攻击结果**的增强点。

结合 Gate 1~5 的结论，当前最合理的增强方向是：

1. 针对 `VMA / IMA` 的**敏感 token 定向硬化**
2. 配套给出**发布策略建议**

其中本轮实现的核心增强点是：

> **Sensitive-Token Targeted Hardening**  
> 只对敏感 token 集合施加更强 embedding/head 噪声，而不把强噪声施加到整张词表。

---

## 2. 为什么选这个方向

Gate 5 已经说明：

- 在全局调大 `alpha_e / alpha_h` 时，
  - 小噪声几乎不带来安全收益；
  - 大噪声会明显破坏正确性。

这说明当前问题不是“全局噪声不够大”，而是：

> 需要一种**更有针对性**的噪声策略，把扰动集中用在更值得保护的 token 上。

如果这个方向有效，理想结果应当是：

- 默认正确性保持稳定；
- 敏感 token 的 `VMA / IMA` 恢复率下降；
- 整体 top1 不一定大幅变化，但 sensitive recovery 应明显改善。

---

## 3. 本轮增强点

### Enhancement A：敏感 token 定向加噪

做法：

- 基于 `Stage J` 路线导出新 artifact；
- 只对敏感 token id 子集施加 embedding/head noise；
- 非敏感 token 保持原工作点；
- 先测试两档：
  - `targeted_mild`
  - `targeted_strong`

### Enhancement B：发布建议（文档层）

不做代码改造，只输出建议：

- 外部发布不应默认附带 `stable_reference`
- 若要发布部署线，应附带更严格的风险说明
- 对特定业务敏感词表可单独硬化

---

## 4. Gate 6 验收标准

- [ ] 至少一个增强 artifact 成功导出
- [ ] 有默认 prompt 正确性结果
- [ ] 有 `VMA` 对比结果
- [ ] 有 `IMA` 对比结果
- [ ] 有 `sensitive_token_recovery_rate` 对比
- [ ] 有 Gate 6 文档沉淀

---

## 5. 一句话结论

> Gate 6 的核心不是继续盲目增大全局噪声，而是验证“把扰动集中施加到敏感 token 子集”是否能在不明显破坏默认正确性的情况下，实质降低 `VMA / IMA` 对敏感 token 的恢复率。 
