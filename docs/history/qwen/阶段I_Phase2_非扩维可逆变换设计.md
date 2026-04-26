# 阶段 I / Phase 2：非扩维可逆变换设计

本文档定义一个新的 **Phase 2 设计方向**：放弃当前会把 hidden 从 `d` 扩到 `d+2h` 的扩维 KeyMat 作为 vLLM / 标准 HF checkpoint 的直接目标，转而设计一类 **不改变原始 hidden shape 的可逆变换**，为后续把更多混淆真正物化为标准权重铺路。

这不是对阶段 F/G/H 的否定，而是对阶段 I 当前实际阻塞点的响应：

> 当前 Stage H 虽然在功能上已经稳定，但它依赖的 Algorithm-1 KeyMat 是扩维变换，因此 embedding/head、attention、FFN、RMSNorm 的参数 shape 都已经脱离原始 Qwen2 标准 checkpoint，无法直接落到标准 HF/vLLM 形态。

---

## 1. 设计目标

新的非扩维方案必须同时满足以下 4 条：

1. **不改变 hidden size**
   - 所有 hidden-side 权重仍保持原始 Qwen2 形状，例如：
     - embedding：`[V, d]`
     - q_proj：`[d_q, d]`
     - o_proj：`[d, d_q]`
     - gate/up/down：`[..., d] / [d, ...]`
2. **可逆**
   - 必须存在 `P` 与 `Q`，满足 `P @ Q = I`
3. **可被标准权重表达**
   - 变换要能直接吸收到 embedding/head、attention、FFN、RMSNorm 的参数里
4. **尽量不破坏标准算子语义**
   - 尤其是 RMSNorm 必须仍然是“标准 RMSNorm + 一组 weight”，而不是阶段 G 那种 metric-matrix 语义

这 4 条一起决定：Phase 2 新方案不能简单把“当前扩维 KeyMat 改成 square matrix”就结束，而必须选一类 **同时兼容 standard RMSNorm 的 square invertible transform**。

---

## 2. 为什么当前扩维 KeyMat 不能直接进入标准 HF / vLLM

阶段 I 的 probe 已经给出直接证据：

- baseline hidden size：`896`
- 当前 KeyMat expanded size：`1152`

对应 `outputs/stage_i/phase2_probe.json` 中可以看到：

- `embedding_weight`：`[151936, 1152] -> [151936, 896]`
- `lm_head_weight`：`[151936, 1152] -> [151936, 896]`
- `q_proj_weight`：`[896, 1152] -> [896, 896]`
- `gate_proj_weight`：`[4864, 1152] -> [4864, 896]`
- `input_layernorm_metric_matrix`：`[1152, 1152] -> [896]`

这说明当前 Phase 2 的真正障碍有两层：

### 2.1 参数 shape 不一致

只要 hidden 被扩到 `d+2h`，就不可能直接把 Stage H 的权重写回到“原始 Qwen2 标准 HF checkpoint”。

### 2.2 RMSNorm 语义不再标准

阶段 G 的 fused RMSNorm 本质上依赖一个 `metric_matrix`：

- 它不再是标准 RMSNorm 的“逐通道 weight 向量”
- 因此即使 shape 勉强对齐，也不能直接交给 vLLM 的标准 RMSNorm kernel

所以如果目标仍然是“标准 HF/vLLM checkpoint”，Phase 2 不应该在当前扩维 KeyMat 上继续缝补，而应该更换 hidden transform 家族。

---

## 3. 推荐的新 hidden transform 家族

### 3.1 结论先说

如果目标是：

- 保持原始 hidden shape
- 保持标准 RMSNorm
- 尽量多把混淆吸收到标准权重

那么推荐的 hidden transform 家族是：

> **Monomial square transform（单项式方阵变换）**

也就是：

\[
P = c \cdot \Pi \cdot S_{\pm}
\]

其中：

- `c`：可选的全局标量（默认先取 `1.0`）
- `Π`：hidden 维 permutation matrix
- `S_{±}`：对角 sign matrix（每个通道取 `+1/-1`）

逆矩阵就是：

\[
Q = P^{-1} = \frac{1}{c} \cdot S_{\pm} \cdot \Pi^{-1}
\]

因为：

- `Π^{-1} = Π^T`
- `S_{±}^{-1} = S_{±}`

---

## 4. 为什么推荐 monomial square transform

### 4.1 对 embedding/head 完全兼容

对于 embedding/head：

\[
\widetilde W_{embed} = W_{embed}\,P
\]

\[
\widetilde W_{head} = W_{head}\,Q^T
\]

由于 `P/Q` 都是 `d x d`，shape 不变，可以直接写回标准 checkpoint。

### 4.2 对 attention / FFN 完全 shape 兼容

例如 attention：

\[
\widetilde W_q = W_q Q^T
\]

\[
\widetilde W_o = P^T W_o
\]

FFN 同理：

- gate/up 右侧乘 `Q^T`
- down 左侧乘 `P^T`

因此所有线性层的 shape 都保持不变。

### 4.3 对标准 RMSNorm 仍可精确表达

这是最关键的一点。

标准 RMSNorm 的输出形式是：

\[
\mathrm{RMSNorm}(x; w)=\frac{x}{\mathrm{RMS}(x)}\odot w
\]

若要在 obfuscated basis `y = xP` 下仍然保持**标准 RMSNorm 语义**，则要求：

\[
\widetilde w
\]

仍然是一个 **逐通道 weight 向量**，而不是矩阵。

只有当 `P` 是 monomial matrix（即 permutation × sign × 全局常数）时，

\[
P^{-1} \operatorname{Diag}(w) P
\]

仍然是对角矩阵，对应一个新的 weight 向量。

因此：

- 一般的 square dense invertible matrix：**不行**
- 一般的正交旋转矩阵：**不行**
- 任意非均匀 diagonal scaling：**通常也不适合直接要求 exact standard RMSNorm**
- permutation + sign + 全局常数：**可以**

这就是为什么 Phase 2 如果坚持“标准 HF/vLLM checkpoint”，推荐 transform family 必须收敛到 monomial 类型。

---

## 5. 新方案与当前阶段资产的关系

这个新方案并不是从头重来，而是对现有 A–H 主线做一次“适配标准部署”的裁剪。

### 5.1 可直接复用的部分

- Stage A 的 `perm_vocab / inv_perm_vocab`
- Stage E/H 的复杂 attention 结构：
  - `R̂_qk`
  - `Ĥ_qk`
  - `Ẑ_block`
  - `τ_kv`
  - `τ_group`
- Stage C/G 的 FFN 中间维：
  - `Z_ffn`
  - `H_ffn`
- Stage I 已经做好的：
  - 标准 HF checkpoint 导出
  - client/server secret 与回归链路

### 5.2 需要替换或弱化的部分

- **替换** 当前扩维 KeyMat
  - 不再用 `d -> d+2h`
  - 改用 `d -> d`
- **弱化** RMSNorm 的设计目标
  - 不再追求“任意可逆 transform 都能融进去”
  - 而是反过来要求 transform 家族满足标准 RMSNorm 的可表达性

这一步会牺牲一部分“变换空间强度”，但换来：

- 标准 checkpoint shape
- 标准 RMSNorm kernel
- 更接近 vLLM 真正可加载的模型形态

---

## 6. 推荐的 Phase 2 新技术路线

### Phase 2-A：先做 square monomial hidden transform

实现一个新的 hidden transform：

- `perm_hidden`
- `sign_hidden`
- 可选全局 `scale_c`

默认建议：

- `scale_c = 1.0`
- 先只做 permutation + sign

原因：

- 最稳定
- RMSNorm exact
- 便于直接验证标准 checkpoint 写回

### Phase 2-B：先打通 embed/head

验证：

- shape 完全一致
- export 后 HF 回归通过
- special/additional/tail token 规则不破坏

### Phase 2-C：再打通 attention / FFN

此时 hidden 侧不再扩维，attention / FFN 只需：

- 吸收新的 square hidden transform
- 保留现有阶段 E/H 已稳定的 intra-head / inter-head / FFN intermediate 结构

这一阶段的目标是：

- `q/k/v/o` 仍然都是标准 shape
- `gate/up/down` 仍然都是标准 shape
- runtime 不再需要扩维 bridge

### Phase 2-D：最后处理 norm

由于 transform 已限定为 monomial family，norm 可以退化成：

- weight permutation
- sign 在对角共轭中抵消
- 全局常数由 RMS normalization 自动消去或单独规范化

这一阶段理论上应能保持**标准 RMSNorm 算子不变**。

---

## 7. 与安全性目标的关系

这个新方向的重点是 **部署可行性**，不是立刻追求最强混淆强度。

它与当前扩维 KeyMat 的关系应这样理解：

- 扩维 KeyMat：更接近论文 Algorithm 1 的表达力，但不适合直接落成标准 HF/vLLM checkpoint
- non-expanding monomial transform：表达力更保守，但更适合作为 **Phase 2 标准部署版本**

因此，后续可以同时保留两条线：

1. **研究线**
   - 保留扩维 KeyMat，继续服务于研究复现与攻击评估
2. **部署线**
   - 引入 non-expanding transform，服务于标准 HF/vLLM checkpoint

---

## 8. 建议的实现顺序

下一轮如果开始真正实现，建议顺序如下：

1. **新增 square hidden transform generator**
   - 只生成 `d x d` 的 monomial transform
2. **先做 embed/head-only checkpoint**
   - 验证导出/加载/回归
3. **再做 block0**
   - attention / FFN / norm 全部保持标准 shape
4. **再做 prefix-2**
   - 观察多层传播是否仍可控
5. **最后再尝试 full-layer / vLLM**

---

## 9. 一句话结论

> 如果 Phase 2 的目标是“继续把混淆写回标准 HF/vLLM checkpoint”，那么下一步不应继续沿当前扩维 KeyMat 缝补，而应切换到一类 **不改变 hidden shape、且对标准 RMSNorm 可精确表达的 monomial square transform**。
