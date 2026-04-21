# 阶段 J：论文一致部署路线说明

## 1. 为什么当前目标需要重定义

到目前为止，仓库里已经建立了两条 `Stage J` 后段路线：

- `buffered redesign line`
  - `artifacts/stage_j_qwen_redesign`
  - 安全面上已经明显优于旧 `stage_j_full_square`
- `standard-visible bridge line`
  - `artifacts/stage_j_qwen_redesign_standard`
  - 已经做到标准键可见、并且可被标准 Hugging Face 加载器加载

但如果目标是：

> **与原始论文一致的混淆部署**

那么当前这两条线都还不是最终答案。

## 2. 核心判断

### 2.1 buffered redesign line 不是最终部署目标

当前 `buffered redesign line` 的关键特征是：

- `embed` 宽度为 `1152`
- `q/o/gate/down` 等权重围绕 `1152` 构造
- `server/model.safetensors` 仍以 `buffer::stage_a_model.*` 键为主

这条线很有价值，因为它已经证明：

- 旧部署线最危险的 `VMA` 结构恢复面可以被显著压下去
- `ISA hidden_state` 也可明显回到接近 `Stage H` 的水平

但它并不等于论文意义上的最终部署线，因为它仍然依赖一条带有非标准内部表达痕迹的导出路径。

### 2.2 standard-visible bridge line 也不是最终部署目标

当前 `standard-visible bridge line` 的价值是：

- 已经完成标准 `model.* / lm_head.*` 键布局导出
- 已经完成标准加载器可加载验证

但它的问题同样明确：

- `bridge_regression` 误差很大
- `generated_ids_exact_match_rate = 0.0`
- 它和 buffered redesign line 仍然强烈不等价

所以：

> 它是一个“标准导出工程通路”，不是一个“论文一致部署语义已经成立”的结论。

## 3. 为什么继续优化当前 bridge 不是最优方向

如果继续只围着 bridge regression 做局部修补，会有两个风险：

1. 容易把“让 bridge 更像 buffered redesign”误当成最终目标；
2. 即便 bridge 渐渐逼近 buffered redesign，也不等于它已经符合论文部署适配思想。

因为论文真正要求的是：

- 不改模型结构
- 在线推理图仍然是标准 Transformer 图
- 复杂扰动通过离线参数重写吸收到标准组件中
- attention / FFN / norm 的关键表达尽量保留

所以真正正确的目标不是：

- “先做一个非标准内部表达，再逼一个 bridge 去模仿它”

而是：

- “直接构造 paper-consistent standard deployable obfuscated checkpoint”

## 4. 新的目标定义

从现在开始，若以“论文一致部署”为目标，`Stage J` 的最终目标应改成：

> **直接输出一条标准运行图、标准键布局、但仍保留论文允许的 attention / FFN / norm 扰动表达的部署线。**

这意味着后续要优先回答的问题是：

### 4.1 哪些表达必须原生进入标准导出线

- embed/head noise + permutation + key-matrix side relation
- attention profile 中真正关键的 q/k 侧与 head/group 侧表达
- FFN 的 component-specific transform
- norm 的 κ 修正逻辑

### 4.2 哪些表达不能继续靠 buffered state 间接保留

- 只存在于 `buffer::stage_a_model.*` 的内部信息
- 只在 custom wrapper / custom module 中可见、但标准权重里不可见的表达

当前 `norm gap` 审计已经说明：

- `offdiag_ratio ≈ 0.972`
- `standard_rmsnorm_equivalent = false`

因此，当前 buffered redesign 里的 metric-based norm 不能被简单视为“以后再抄成一个标准 RMSNorm 权重向量”。

## 5. 新的推进策略

因此，后续 `Stage J` 的正确推进顺序应改为：

1. 把当前 bridge 和 buffered redesign 的差距保留为参考，而不是继续把它当最终目标
2. 逐项定义 paper-consistent deployment 所需的最小表达集合
3. 直接面向标准 `model.*` 权重键布局实现这些表达
4. 再重新跑 correctness / `VMA / IMA / ISA`

## 6. 一句话结论

> 如果目标是“与原始论文一致的混淆部署”，那么当前 buffered redesign 与 standard-visible bridge 都只能算过渡形态；下一步不该继续把“bridge 更像 buffered redesign”当成最终目标，而应直接转向构造一条 paper-consistent、标准运行图、标准键布局、且尽量保留论文复杂扰动表达的真正部署线。 
