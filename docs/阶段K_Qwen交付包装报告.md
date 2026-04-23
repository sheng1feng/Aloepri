# 阶段 K：Qwen 交付包装报告

> Canonical note: 本文档只回答当前 `Stage K` 的状态，不承担全局主线说明。Qwen 唯一主线入口见 [docs/论文一致最终部署主线.md](论文一致最终部署主线.md)。

## 1. 新阶段目标

新版 `Stage K` 不再默认包装旧 `stage_j_full_square*`，而是包装：

- 新版 `Stage J` 物化出的 `artifacts/stage_j_qwen_redesign`

它的职责是：

- 提供 release catalog
- 提供 profile 命名
- 固定 client/server contract
- 暴露统一推理入口

## 2. 当前包装策略

当前新版 `Stage K` 先使用两个 release profile 名称：

- `stable_reference`
- `tiny_a`

两者当前都指向 redesigned `Stage J` artifact，但语义上分别承担：

- baseline / packaging verification
- default delivery alias

后续如果 redesigned Stage J 继续分化出不同 noise/workpoint，可在不改 release 结构的前提下替换 profile source。

## 3. 与 legacy Stage K 的关系

旧版 `Stage K` 仍然保留，但它包装的是 conservative standard-shape line，不再是新版 `Stage K` 的唯一语义。
