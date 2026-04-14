# 阶段 H-K 论文对齐检查表

## 1. 使用方式

这张表不回答“新线已经彻底正确了吗”，而是回答：

> 当前 redesigned `Stage H-K` 线，和论文部署适配机制相比，哪些表达已经保留，哪些只是 bootstrap source 中存在，哪些已经被 export / release 证明，哪些仍未被证明。

## 2. 检查表

| 项目 | 论文要求 | 当前状态 | 备注 |
|---|---|---|---|
| embed/head noise | 需要保留 | `部分已证实` | `Stage H` source config 与 `Stage J` manifest 已记录 `alpha_e / alpha_h` |
| token permutation | 需要保留 | `已证实` | client secret 与现有安全脚本仍依赖 permutation |
| key matrix | 需要保留 | `已证实` | `lambda=0.3, h=128` 在 source config 与审计输出中可见 |
| attention rotation/scaling/profile | 尽量保留 | `source 中已证实` | `attention_profile=rqk_hqk_block_taukv_taugroup` 已在 `Stage H` source 与 `Stage J` manifest 中可见 |
| attention block/head/group diversity | 尽量保留 | `安全结果侧已间接支持` | redesign `VMA` 已显著贴近 `Stage H`，但 export 级 component proof 仍不完整 |
| FFN component transform | 尽量保留 | `未充分证实` | 当前 manifest 没有单独列出 gate/up/down 的 export 级证明 |
| norm kappa correction | 尽量保留 | `未充分证实` | `Stage H` inventory 保留该语义，但 `Stage J` export 级证明仍不足 |
| standard runtime graph | 必须保留 | `已证实` | boundary audit 当前仍声明运行时不需要自定义在线算子 |
| 直接可被标准权重读取 | 最好成立 | `未证实` | redesign artifact 仍使用 buffered stage-style safetensor 键，不是旧式标准 `model.*` 键 |
| key-matrix expansion boundary | 允许存在 | `已证实` | redesign `IMA` loader 读到的 embedding 维度大于 baseline，说明当前线并未退回旧 non-expanding line |

## 3. 当前最重要结论

### 3.1 已经被证实的部分

- redesign 线不是旧 `full_square` 的换名版本
- `VMA` 上 redesign `Stage J` 已经从旧 `Stage J stable` 的高恢复率降到与 `Stage H` 同级
- `ISA hidden_state` 上 redesign `Stage J` 也已经贴近 `Stage H`

### 3.2 尚未被证实的部分

- `Stage J` export 仍未把 attention / FFN / norm 表达证明成标准 `model.*` 权重键可直接读取的形式
- `FFN` 与 `norm` 的 component-level export 证明仍然缺失
- `IMA` 上虽然 cosine 已贴近 `Stage H`，但 top1 恢复率仍然很高

## 4. 当前建议

如果继续推进论文对齐，优先级应是：

1. 补 `Stage J` export 的 component-level expression proof
2. 明确 `FFN / norm` 在 export 级别的保留证据
3. 再决定是否把 redesign 线继续收缩到标准权重键布局，或保留当前 buffered stage-style 布局
