# artificial-scientist-lab/Learn2Design-2026 — NeurIPS 2026 引力波探测器设计竞赛

> 仓库来源归档（ingest）

- **类型：** repo / competition / benchmark / gravitational-wave / optimization / jax
- **URL：** <https://github.com/artificial-scientist-lab/Learn2Design-2026>
- **许可：** **MIT**
- **提交门户：** <https://submit.learn2design2026.com/competitions/4/>
- **项目站：** <https://www.learn2design2026.com/>
- **入库日期：** 2026-09-12
- **一句话说明：** NeurIPS 2026 Challenge 官方仓：starter kit、UIFO 搜索空间文档、Round 1 全量评测数据（43 队 × 10 拓扑）、提交规范与 H100 评测 VM 说明。

## 维护者整理的结构化入口（摘自 README）

| 主题 | 入口 |
|------|------|
| 竞赛提案 PDF | `Learn2Design_details.pdf` |
| 提交规范 | `docs/submission.md` |
| 评测硬件 | `docs/submission.md#evaluation-hardware`（标准 H100 VM） |
| dfbench 概览 | `docs/dfbench_overview.md` |
| Round 1 数据 | `competition_data/round1/`（每队 10 次得分、收敛检查点、效率统计） |
| 依赖仿真 | [Differometor](https://github.com/artificial-scientist-lab/Differometor) · [Differometor-Benchmark](https://github.com/artificial-scientist-lab/Differometor-Benchmark) |

## 评测协议要点

- 参赛方提交 **算法 ZIP**，非最终光机参数表。
- 每拓扑 **4h** 预算；取 **约束可行** 的最优损失；若无可行解则回退随机搜索基线。
- 分数 = 10 隐藏拓扑损失 **算术平均**（越低越好）。
- Round 1 已评 43 提交 + 7 条组织者基线；公开 leaderboard 图见 `media/round1_leaderboard.png`。

## 对 wiki 的映射

- [`wiki/entities/paper-designing-physics-experiments-with-ai.md`](../../wiki/entities/paper-designing-physics-experiments-with-ai.md)
- [`sources/sites/learn2design_2026.md`](../sites/learn2design_2026.md)
- [`sources/repos/artificial_scientist_lab_differometor.md`](artificial_scientist_lab_differometor.md)

## 当前提炼状态

- [x] 步骤 2.5：**已开源**（MIT + 公开 Round 1 数据）
