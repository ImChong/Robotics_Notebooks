---
type: entity
tags: [robot, hardware, humanoid, quadruped, industry]
status: complete
updated: 2026-04-21
related:
  - ./humanoid-robot.md
  - ../tasks/locomotion.md
  - ../concepts/whole-body-control.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "Boston Dynamics 是全球足式机器人的领军企业，旗下的 Atlas 和 Spot 分别定义了人形与四足机器人的最高动态性能标准。其基于解析动力学的模型预测控制（MPC）与全身控制（WBC）技术栈至今仍是行业标杆。"
---

# Boston Dynamics（波士顿动力）

**Boston Dynamics** 是一家全球顶尖的机器人工程公司，以其在足式机器人运动控制、平衡和动力学领域的卓越成就而闻名。从 1992 年从 MIT 的 Leg Laboratory 独立至今，它先后推出了 BigDog, Cheetah, LS3, Atlas 和 Spot 等一系列划时代的机器人产品，极大地推动了腿足式机器人（Legged Robotics）从学术理论向实际工程部署的演进。

## 代表性产品

### 1. Atlas (人形机器人)
Atlas 是世界上最先进的人形机器人之一，代表了双足机器人动态性能的巅峰。
- **液压版 (Legacy Hydraulic Atlas)**：拥有 28 个液压关节。借助液压系统极高的功率密度（Power Density），它实现了跑酷、后空翻、跳跃等人类级别的灵巧和高爆发力动作。在控制层面，液压版 Atlas 采用了极其先进的 **全身控制 (Whole-Body Control, WBC)** 和 **模型预测控制 (Model Predictive Control, MPC)**。通过在极短的控制周期内求解带有接触力约束的二次规划（QP）问题，Atlas 展现出了惊人的扰动恢复能力。
- **全电版 (All-Electric Atlas)**：2024 年，波士顿动力宣布停产液压版，转向全新的全电驱动 Atlas。全电版弃用了复杂的液压泵和管路，转向基于高转矩密度电机的设计。全电版的设计更加极简，关节的旋转范围甚至超越了人类关节的物理极限（例如大腿可以 360 度旋转），这预示着人形机器人从“证明动态能力的实验室原型”向“通用多功能工具”的商业化转型。同时，全电版更利于机器学习算法（如 RL 和 VLA）的端到端部署。

### 2. Spot (四足机器人)
Spot 是全球最成功的商业化足式机器人，广泛应用于工业巡检、测绘和高危环境作业。
- **环境适应性**：其核心竞争力在于极强的环境适应性（如 AutoWalk 自主巡检系统）和极其稳定的步态控制。
- **商业化证明**：Spot 证明了四足机器人能够在工厂、矿区、变电站等结构复杂、障碍众多的真实场景中，以极高的可靠性（MTBF，平均故障间隔时间）连续运行。它不仅是一个硬件平台，更是一个集成了避障、楼梯攀爬、感知与导航的完整机器人产品。

### 3. Stretch (仓储机器人)
Stretch 是一款针对物流搬运设计的非类人机器人。它拥有一个巨大的多自由度机械臂和一个带有吸盘的末端执行器，安装在全向移动底座上。Stretch 展示了波士顿动力将过去在足式平衡中积累的“质心控制”与“动态操作”技术，成功降维并迁移到工业垂直领域的商业能力。

## 核心技术路线与工程哲学

1. **基于优化的控制 (Optimization-based Control)**：Boston Dynamics 在过去二十年的长项在于基于解析动力学模型的最优控制算法。Atlas 的平稳运动很大程度上归功于其对 WBC 和接触力摩擦锥（Friction Cone）优化的极致压榨。他们能够在 1000Hz 的频率下实时求解复杂的机器人动力学方程。
2. **高功率密度执行器 (High Power Density Actuators)**：无论是早期的液压驱动器，还是现在的定制全电伺服电机，波士顿动力在执行器硬件和传动机制上的积累始终处于行业第一梯队。
3. **感知与规划集成 (Perception-Action Integration)**：其机器人具备极强的实时环境感知能力，能够通过深度相机和激光雷达动态识别地形（Terrain Adaptation），并在线生成安全的足迹规划（Footstep Planning）与质心轨迹。

## 行业地位与未来挑战

如果说 Unitree 等公司通过供应链优势推动了足式机器人的“平民化”和“普及化”，那么 Boston Dynamics 则始终在探索“**机器人动态性能的物理极限**”。

在深度学习和强化学习（RL）爆发的今天，波士顿动力经典的“模型驱动（Model-based）”路线面临着数据驱动（Data-driven）路线的挑战。虽然其在人工智能（特别是大模型和模仿学习端到端控制）方面的 PR 相对低调，但其底层极其扎实的运控算法和硬件平台，仍是当前所有 RL 从业者试图超越的物理标杆。

## 关联页面
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 参考来源
- Boston Dynamics 官方网站与技术博客。
- Kuindersma, S., et al. (2016). *Optimization-based locomotion planning, estimation, and control design for the atlas humanoid*.
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
