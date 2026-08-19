# H2R-Bench（arXiv:2608.13049）

> 来源归档（ingest）

- **标题：** H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models
- **类型：** paper / world-model / cross-embodiment / manipulation / benchmark
- **arXiv：** <https://arxiv.org/abs/2608.13049>
- **项目页：** <https://rongdingyi.github.io/H2R-Bench/>（归档见 [`sources/sites/h2r-bench.md`](../sites/h2r-bench.md)）
- **代码：** <https://github.com/Rongdingyi/H2R-Bench>（归档见 [`sources/repos/h2r-bench.md`](../repos/h2r-bench.md)）
- **入库日期：** 2026-08-19
- **一句话说明：** 把「人类第一视角操作视频能否变成指定机器人本体的训练素材」做成五维可诊断基准；评测 11 个视频生成模型 × 6 类操作 × 2 种机器人本体。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-19）：** 有 Paper / Project 链；GitHub 仓主要为 `docs/` 项目页源码。
- **仓库：** README 写明 **evaluation code and benchmark annotations 尚未发布**；HuggingFace 仅有 papers 页，无 benchmark 数据。
- **结论：** **部分开源**（项目页 + 仓骨架；核心评测与标注待发布）。

## 摘录：评测维度

| 维度 | 含义 |
|------|------|
| 目标状态 | 操作后物体/场景是否到达期望状态 |
| 动作事件 | 关键 manipulation 事件是否出现 |
| 功能接触 | 功能相关接触是否合理 |
| 本体正确性 | 生成视频是否符合目标机器人本体 |
| 视频质量 | 时序一致性与视觉保真 |

**对 wiki 的映射：** [`wiki/entities/paper-h2r-bench.md`](../../wiki/entities/paper-h2r-bench.md)；交叉 [生成式世界模型](../../wiki/methods/generative-world-models.md)、[跨本体迁移策略](../../wiki/queries/cross-embodiment-transfer-strategy.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查
