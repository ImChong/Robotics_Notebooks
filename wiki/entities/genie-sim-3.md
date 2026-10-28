---
type: entity
tags: [agibot, simulation, benchmark, sim2real, open-source]
status: complete
updated: 2026-10-28
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../overview/agibot-june-2026-release-technology-map.md
  - ../overview/agibot-release-category-02-sim-training-eval.md
  - ./robocoliseum.md
  - ./go-2.md
  - ../queries/simulator-selection-guide.md
  - ../overview/robot-training-stack-layers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
  - ../../sources/repos/genie_sim.md
  - ../../sources/sites/genie_sim.md
  - ../../sources/repos/genie_sim_3.md
  - ../../sources/repos/genie_sim_robocoliseum.md
  - ../../sources/sites/robocoliseum.md
summary: "Genie Sim 3.0 是智元开源的仿真训练与评测平台：支持自然语言/图像生成可交互三维场景，提供 Genie Sim Benchmark 五类能力评测，并对接 RLinf、并行仿真与在线微调。"
---

# Genie Sim 3.0

**Genie Sim 3.0** 是智元 [AgibotTech/genie_sim](https://github.com/AgibotTech/genie_sim) 开源的 **高保真综合仿真平台**（入口：<https://agibot-world.com/genie-sim>）。在 [智元 2026-06 发布地图](../overview/agibot-june-2026-release-technology-map.md) 中承担 **仿真训练与评测** 段：把 **场景生成** 接到训练、评测与后训练，以降低真机试错成本。

## 一句话定义

**语言/图像驱动的三维场景生成 + 机器人能力 Benchmark + RL 训练接口**，目标是把仿真价值落到 **减少真机试错** 而非仅做可视化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| Gym | OpenAI Gym API | 强化学习标准环境接口惯例 |

## 为什么重要

- **场景生成：** 一句话或一张图生成可漫游、可训练的三维世界，并输出 RGB、深度、激光雷达等多模态数据。
- **Genie Sim Benchmark：** 五类核心能力——指令跟随、空间理解、操作执行、扰动适应、Sim2Real 迁移。
- **RoboColiseum 在线挑战赛：** [RoboColiseum](./robocoliseum.md) 把仿真评测产品化为 **四块独立能力榜**（instruction / spatial / manip / robust）+ **WebSocket 隧道远程推理** + ModelScope **LeRobot v2.1** 训练数据；skill 集成于 genie_sim 开源仓。
- **训练栈：** 支持 RLinf、并行仿真、标准 Gym 接口与 **在线微调**。
- **Sim2Real 叙事：** 材料称相同模型在仿真与真机评测差异 **<10%**（策展文建议等待更多任务验证）。

## 源码架构与运行

依据 [官方源码快照](../../sources/repos/genie_sim.md)，目前先用 `geniesim` CLI 管理容器和工作区，再按目标选择两条**独立的运行栈**：

| 路径 | 用途与边界 |
|------|------------|
| `geniesim_benchmark` | 直接使用 Isaac Sim 运行基准、连接策略推理服务并评分；RoboColiseum 使用同一评测生态。 |
| `geniesim_ros` | ROS 2 实时交互引擎，物理、渲染与机器人共享仿真时间，接入 MoveIt 2、ros2_control、遥操作和记录。 |

官方计划将 benchmark 重构为 `geniesim_ros` 之上的层；截至该快照仍未合并，场景格式和启动流程不能假设互通。场景生成、世界生成、遥操作和自动采集分别由其他模块承担；`geniesim_world` 输出的 Gaussian/PLY 接入实时引擎仍在计划中。

### 实践入口与限制

- 默认容器以 **Isaac Sim 5.1 + ROS 2 Jazzy** 为目标；官方将 Humble 限定在已停止新功能支持的 Isaac Sim 4.5 容器。现有 Humble 系统集成应先检查容器与接口兼容性。
- RT Engine 支持 Isaac Sim PhysX、实验性 Isaac Sim Newton 与独立 Newton；官方说明独立 Newton 路径支持布料和软体。后端能力不能一概而论。
- 仓库公开代码并提供外置资产、数据集入口；README 仍将上传全部资产和数据集列为待办。`geniesim_*` 和 `data_collection` 默认 MPL 2.0，重建模块含多种许可，使用前需逐项核对。
- 官方展示的仿真与真机结果是特定任务、数据量和模型下的实验结果，不能推断所有任务均有同等 Sim2Real 精度。README 宣布 2026-11-02 有重大更新，具体内容应待发布后核实。

## 与 GE-Sim 2.0 / GO-2

| 项目 | 分工 |
|------|------|
| GE-Sim 2.0 | **视频世界模型** 闭环 |
| Genie Sim 3.0 | **场景级仿真** 训练与 benchmark |
| GO-2 | 文内在 Genie Sim Benchmark 与 Sim2Real 评测中报告领先结果 |

## 关联页面

- [仿真训练与评测分类 hub](../overview/agibot-release-category-02-sim-training-eval.md)
- [Paper Notebooks 索引实体](./genie-sim-3.md)
- [训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md)
- [RoboColiseum](./robocoliseum.md) — Genie Sim 在线仿真挑战赛与四榜
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页可归入其 ③ / ④ 层：Genie Sim Benchmark 五类能力仿真评测与 sim2real 对接

## 参考来源

- [Genie Sim 3.0 源码归档](../../sources/repos/genie_sim_3.md)（<https://github.com/AgibotTech/genie_sim>）

- [Genie Sim 3.0 源码归档](../../sources/repos/genie_sim.md)（<https://github.com/AgibotTech/genie_sim>）

- [Genie Sim 项目页归档](../../sources/sites/genie_sim.md)

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Genie Sim 3.0](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Genie_Sim_3.0__A_High-Fidelity_Comprehensive_Simulation_Platform_for_Humanoid_Robot/Genie_Sim_3.0__A_High-Fidelity_Comprehensive_Simulation_Platform_for_Humanoid_Robot.html)
- [Genie Sim GitHub](https://github.com/AgibotTech/genie_sim)
- [RoboColiseum 官网](https://robocoliseum.ai/)
