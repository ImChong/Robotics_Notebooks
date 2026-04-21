---
type: entity
tags: [robot, hardware, humanoid, quadruped, industry]
status: complete
updated: 2026-04-20
related:
  - ./humanoid-robot.md
  - ../tasks/locomotion.md
  - ../concepts/whole-body-control.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "Boston Dynamics 是全球足式机器人的领军企业，旗下的 Atlas 和 Spot 分别定义了人形与四足机器人的最高动态性能标准。"
---

# Boston Dynamics（波士顿动力）

**Boston Dynamics** 是一家全球顶尖的机器人工程公司，以其在足式机器人运动控制、平衡和动力学领域的卓越成就而闻名。从 1992 年从 MIT 独立至今，它先后推出了 BigDog, Cheetah, LS3, Atlas 和 Spot 等一系列划时代的机器人。

## 代表性产品

### 1. Atlas (人形机器人)
- **液压版 (Legacy)**：28 个液压关节，以极高的功率密度实现了跑酷、后空翻等人类级别的灵巧动作。它采用了先进的 **全身控制 (WBC)** 和 **模型预测控制 (MPC)**。
- **全电版 (All-Electric)**：2024 年发布，弃用了复杂的液压系统，转向高转矩电机驱动。设计更极简，关节旋转范围更大，预示着人形机器人从“实验室原型”向“通用工具”的转型。

### 2. Spot (四足机器人)
- 全球最成功的商业化足式机器人。其核心竞争力在于极强的环境适应性（AutoWalk）和稳定的步态控制。它证明了四足机器人可以在工厂、矿区、变电站等复杂真实场景中稳定工作。

### 3. Stretch (仓储机器人)
- 针对物流搬运设计的非类人机器人，拥有巨大的机械臂和移动底座，展示了波士顿动力将足式控制技术迁移到工业垂直领域的商业能力。

## 核心技术路线

1. **基于优化的控制 (Optimization-based)**：Boston Dynamics 的长项在于基于动力学模型的优化算法。Atlas 的平稳运动很大程度上归功于其对 WBC 和接触力优化的极致压榨。
2. **高功率密度执行器**：无论是早期的液压技术还是现在的定制电机，其硬件性能始终处于行业第一梯队。
3. **感知与规划集成**：其机器人具备极强的实时感知能力，能够动态识别地形并自主规划足迹。

## 行业地位

如果说 Unitree 推动了足式机器人的“平民化”，那么 Boston Dynamics 则始终在探索“**动态性能的物理极限**”。虽然在人工智能（特别是大模型集成）方面相对低调，但其底层的运控算法至今仍是行业标杆。

## 关联页面
- [人形机器人](./humanoid-robot.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 参考来源
- Boston Dynamics Official Website & Blog.
- Kuindersma, S., et al. (2016). *Optimization-based locomotion planning, estimation, and control design for the atlas humanoid*.
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
