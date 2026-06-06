---
type: entity
tags: [course, nvidia, isaac, omniverse, openusd, physical-ai, sim2real]
status: complete
updated: 2026-05-21
related:
  - ./nvidia-so101-sim2real-lab-workflow.md
  - ./nvidia-omniverse.md
  - ./isaac-gym-isaac-lab.md
  - ./lerobot.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/sites/nvidia-physical-ai-learning.md
  - ../../sources/courses/nvidia_sim_to_real_so101_isaac.md
summary: "NVIDIA Physical AI Learning 是官方免费自学门户，索引 Isaac、OpenUSD、医疗机器人与 SO-101 操作臂 Sim2Real 等多条动手路径，并对接 Brev 云 GPU 环境。"
---

# NVIDIA Physical AI Learning

**NVIDIA Physical AI Learning** 是 NVIDIA 面向 **Physical AI**（能感知、推理物理关系、执行动作并适应真实环境的 AI 系统）的 **免费自学课程门户**。它把 Isaac Sim/Lab/ROS、OpenUSD 数字孪生、医疗机器人等主题组织成可独立推进的学习路径，并指向云端 GPU（Brev）与 Omniverse 开发者社区。

## 一句话定义

NVIDIA 官方的 Physical AI 自学路径总入口：按主题选课后，用 Isaac 栈与 OpenUSD 把仿真训练接到真机部署。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| AI | Artificial Intelligence | 人工智能 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |

## 为什么重要

- **工程主线清晰：** 门户把「仿真 → 数据 → 策略 → 真机」拆成可跟做的模块，比零散博客更适合作为本知识库的 **厂商官方课程锚点**。
- **与仓库已有实体互补：** 本库已有 [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)、[Omniverse](./nvidia-omniverse.md)、[GR00T-VisualSim2Real](./gr00t-visual-sim2real.md) 等 **研究/产品** 页；本门户页负责 **自学路径选型**，避免把课程目录散落在各实体页脚注里。
- **旗舰动手课：** [SO-101 Sim2Real 实验课](./nvidia-so101-sim2real-lab-workflow.md) 在同一门户下，系统演练四种 sim2real 策略，适合作为 manipulation + VLA 入门实验。

## 门户内的主要学习路径（摘要）

| 路径 | 侧重 | 与本库关联 |
|------|------|------------|
| SO-101 Sim2Real | 操作臂 VLA + 四类 gap 策略 | [nvidia-so101-sim2real-lab-workflow](./nvidia-so101-sim2real-lab-workflow.md) |
| Isaac Sim / Isaac Lab 入门 | 仿真与 GPU RL | [isaac-gym-isaac-lab](./isaac-gym-isaac-lab.md) |
| Omniverse + OpenUSD | 工业数字孪生场景组合 | [nvidia-omniverse](./nvidia-omniverse.md) |
| Isaac ROS | ROS 2 + NITROS 真机感知导航 | [ros2-basics](../concepts/ros2-basics.md) |
| Learn OpenUSD | USD 课纲与认证 | 资产管线、URDF→USD |

无本地 GPU 时，官方推荐通过 **NVIDIA Brev** 启动预配置 Isaac / Omniverse 环境（门户页内 Launchable 链接）。

## 与「生成式 AI」的边界（课程共性）

门户强调 Physical AI 与纯生成式/对话式 AI 的差异：必须 **感知真实传感器、在物理约束下行动、适应环境变化**。这一边界与 [Sim2Real](../concepts/sim2real.md) 讨论中的「仿真分布 ≠ 真机分布」一致，但课程更偏 **动手验证** 而非理论综述。

## 参考来源

- [NVIDIA Physical AI Learning 门户](../../sources/sites/nvidia-physical-ai-learning.md)
- [SO-101 Sim2Real 课程归档](../../sources/courses/nvidia_sim_to_real_so101_isaac.md)
- [What is Physical AI?（NVIDIA Glossary）](https://www.nvidia.com/en-us/glossary/generative-physical-ai/)

## 关联页面

- [NVIDIA SO-101 Sim2Real 实验 workflow](./nvidia-so101-sim2real-lab-workflow.md)
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)
- [NVIDIA Omniverse](./nvidia-omniverse.md)
- [LeRobot](./lerobot.md)
- [Sim2Real](../concepts/sim2real.md)

## 推荐继续阅读

- [Physical AI Learning 门户](https://docs.nvidia.com/learning/physical-ai/)
- [Learn OpenUSD 课纲](https://docs.nvidia.com/learn-openusd/latest/index.html)
- [Query：如何缩小 sim2real gap](../queries/sim2real-gap-reduction.md)
