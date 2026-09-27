# NAVSIM（autonomousvision/navsim）

- **标题：** NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking
- **类型：** repo / benchmark / pseudo-simulation
- **链接：** <https://github.com/autonomousvision/navsim>
- **论文：** <https://arxiv.org/abs/2406.15349>
- **入库日期：** 2026-09-27
- **一句话说明：** 开环 **伪仿真** 驾驶规划基准：用真实 log + 轨迹附近合成观测评估 E2E 规划；提供 **PDMS / EPDMS** 与 navtrain/navtest 划分。

## 开源状态

- **已开源**（Apache-2.0）：主分支为 NAVSIM v2；v1.1 分支仍服务 **navtest**  leaderboard（见仓库 README）。

## 与 MM-Future 的关系

- [MM-Future](../papers/mm_future_arxiv_2609_20377.md) 在 **navtrain** 训练，报告 **v1 navtest PDMS** 与 **v2 navtest EPDMS**。
- 复现需按仓库 `docs/install.md` 准备 OpenScene / nuPlan 系资产与 metric 工具链。

## 沉淀到 wiki

- [paper-mm-future](../../wiki/entities/paper-mm-future.md)
- [paper-rise-adaptive-imagination-wam](../../wiki/entities/paper-rise-adaptive-imagination-wam.md)
