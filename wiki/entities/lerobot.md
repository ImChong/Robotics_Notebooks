---

type: entity
title: LeRobot (Hugging Face)
tags: [framework, robot-learning, open-source, dataset, huggingface]
summary: "LeRobot 是 Hugging Face 开发的具身智能全栈框架，旨在将 Transformers 生态迁移到机器人领域，支持高效数据采集与策略训练。"
updated: 2026-07-11
related:
  - ./paper-evo1-lightweight-vla.md
  - ./openvla.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../methods/vla.md
---

# LeRobot (Hugging Face)

**LeRobot** 是由 Hugging Face 开发并维护的一个**具身智能全栈框架**。它旨在将自然语言处理（NLP）领域的成熟生态（如 `transformers` 库和模型 Hub）迁移到机器人领域，提供从数据采集、策略训练到实物部署的一站式工具。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构，常与 ALOHA 配套 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| CAD | Computer-Aided Design | 计算机辅助设计，硬件结构建模 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |

## 为什么重要？

在具身智能的爆发期，LeRobot 扮演了“机器人届的 Transformers”角色：
- **生态对齐**：通过与 Hugging Face 模型库和数据集库打通，极大降低了开发者共享和复用机器人策略（如 [diffusion-policy](../methods/diffusion-policy.md)）的门槛。
- **开源硬件支持**：原生支持低成本开源硬件（如 Koch 机械臂），推动了“人人皆可机器人”的普及。
- **标准化数据格式**：定义了一套高效、可扩展的具身智能数据存储标准，方便不同团队之间的数据交换。

## 核心组件

- **Dataset Library**：支持加载和上传大规模机器人演示数据集。
- **Policy Library**：内置了多种主流算法（如 ACT、Diffusion Policy、TD-MPC2）。
- **Hardware Interface**：提供了一套简洁的 Python 接口，用于连接电机、传感器和真实机器人。

## 与其他系统的关系

- **上层应用**：[xbotics-embodied-guide](../../sources/repos/xbotics-embodied-guide.md) 将 LeRobot 推荐为实现开源实物部署的核心框架。
- **对比**：相比传统的 [ros2-basics](../concepts/ros2-basics.md)，LeRobot 更侧重于“数据驱动型”的端到端学习，而非复杂的分布式中间件逻辑。
- **互补 I/O 栈**：[RIO（Robot I/O）](./robot-io-rio.md) 侧重 **本机实时闭环** 与可切换中间件上的 **异步策略推理**；官方文档叙述可 **导出到 LeRobot / DROID 等格式** 再进入常见训练管线，二者常在「采集/部署」与「数据集/训练」两侧分工。
- **NVIDIA 官方课：** [SO-101 Sim2Real 实验 workflow](./nvidia-so101-sim2real-lab-workflow.md) 用 `lerobot-record`（`so101_follower` / `so101_leader`）采集真机少量演示，并与 Isaac Lab 仿真演示做 Co-training。
- **整机项目协作：** [Tnkr](./tnkr.md) 侧重把 CAD、线束、代码版本与部署/运行数据收进同一开源项目仓库；训练侧仍常导出到 LeRobot 等数据集格式，二者分工不同。
- **ROBOTIS 全栈集成：** [Cyclo Intelligence](./cyclo-intelligence.md) 以子模块钉版本集成 LeRobot，在 Docker 策略容器内提供 ACT/SmolVLA/π₀ 等推理后端，并由行为树编排 `LOAD/RESUME/STOP` 生命周期。
- **轻量 VLA 官方集成：** [Evo-1](./paper-evo1-lightweight-vla.md)（MINT-SJTU，CVPR 2026）已并入 **官方 LeRobot 主仓**；SO100/SO101 可用 `lerobot-record --policy.path=<Evo-1 checkpoint>` 闭环，训练侧数据格式为 **LeRobot v2.1**。
- **部署/Agent OS 对照：** [DimOS（Dimensional）](./dimensionalos-dimos.md) 侧重 **现场 Module 编排、SLAM 导航、空间记忆与 MCP 自然语言控制**；与 LeRobot 的 **数据集 Hub + 策略训练** 正交，常在「训练用 LeRobot、集成用 DimOS/ROS」分层共存。

## 参考来源
- [LeRobot 仓库归档](../../sources/repos/lerobot.md) — 本批导航/SLAM 栈 ingest 同步的官方 GitHub source
- [NVIDIA SO-101 Sim2Real 课程](../../sources/courses/nvidia_sim_to_real_so101_isaac.md) — `lerobot-record` 采集 so101_follower/leader 真机与仿真演示
- [Xbotics-Embodied-Guide](../../sources/repos/xbotics-embodied-guide.md)
- [RIO 仓库与论文归档](../../sources/repos/robot-io-rio.md) — 与 LeRobot 数据导出衔接的跨形态实时 I/O 框架（对照阅读）
- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)
- [Cyclo Intelligence 仓库归档](../../sources/repos/cyclo_intelligence.md) — LeRobot 作为 Cyclo 推理后端之一
- [Evo-1 论文与仓库归档](../../sources/papers/evo1_arxiv_2511_04555.md) — 官方 LeRobot 内置轻量 VLA 策略（SO100/SO101）
