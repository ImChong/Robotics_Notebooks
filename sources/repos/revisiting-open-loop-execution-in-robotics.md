# Revisiting Open-Loop Execution in Robotics 官方 GitHub 总览仓

- **类型：** 官方 GitHub 仓库（论文索引 README 仓）
- **地址：** <https://github.com/Michaelszeng/Revisiting-Open-Loop-Execution-in-Robotics>
- **项目页：** <https://revisiting-open-loop-action-chunking.github.io/>
- **论文：** arXiv:2608.15938v2 — [`sources/papers/revisiting_open_loop_action_chunking_arxiv_2608_15938.md`](../papers/revisiting_open_loop_action_chunking_arxiv_2608_15938.md)
- **入库日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 核对官方仓库 `LICENSE`（MIT）与 `README.md` 全部内容及所列子仓库/数据集索引，确认本仓本身不含可运行源码（0 个 `.py` 文件）
- **一句话说明：** 论文的 GitHub 导航索引仓，目前仅包含 `LICENSE` 与 `README.md`，通过外链汇总 6 个分基准/分任务子仓库与 2 个 HuggingFace 数据集；真机硬件推理代码不开源。

## 开源核查（步骤 2.5，2026-11-19）

| 项 | 状态 |
|----|------|
| 仓库内容 | **仅 `LICENSE`（MIT）+ `README.md`**，仓内 **0 个 `.py` 代码文件**，无统一训练/推理入口 |
| 仿真与训练子仓（外链 6 个） | `README.md` 分别外链 FurnitureSim、PushT-M、PushT-D、GearInsertion、Kitchen 与策略基线等 6 个子仓 |
| 数据集（外链 2 个） | `README.md` 索引 2 个 HuggingFace 数据集 |
| 真机推理栈 | **未开源** — README 明确说明真机硬件/推理代码为内部专有实现，不公开 |
| 结论 | **主索引仓本身无直接可运行源码**；wiki 实体页 `## 源码运行时序图` 标为 **不适用**，工程评估时须明确区分「索引仓 README 与分散子仓」和「未公开的真机推理栈」 |

## 对 wiki 的映射

- 关联实体页：[Revisiting Open-Loop Action Chunking（论文实体）](../../wiki/entities/paper-revisiting-open-loop-action-chunking.md)
- 项目页归档：[项目页摘要](../sites/revisiting-open-loop-action-chunking.md)
