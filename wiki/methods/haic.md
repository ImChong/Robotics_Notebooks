---
type: method
tags: [rl, world-model, teacher-student, imitation-learning, sim2real]
status: complete
updated: 2026-04-27
related:
  - ./model-based-rl.md
  - ./amp-reward.md
  - ../concepts/privileged-training.md
sources:
  - ../../sources/papers/motion_control_projects.md
summary: "HAIC (Hierarchical AI Controller) 提出了一种结合世界模型的两阶段教师-学生训练范式，用于解决物体交互任务中的感知-动作协调难题。"
---

# HAIC: 基于世界模型的教师-学生训练

在复杂的物体交互任务（如搬运、协作、精细操作）中，机器人不仅要模仿姿态，还要实时预测物体状态和外力。**HAIC** 提出了一种创新的训练范式，通过世界模型（World Model）将特权信息（Privileged Info）有效地转化为可部署的学生策略。

## 两阶段训练范式

HAIC 的核心在于如何平滑地从“全知全能”的教师过渡到“仅靠观测”的学生。

### 第一阶段：联合预训练 (Joint Pre-training)
- **教师策略 (Teacher Policy)**：输入特权信息（物体精确位姿、外力、质量等），输出专家动作和价值估计。
- **世界模型 (World Model)**：**关键枢纽**。它的目标是输入可观测的本体感知历史，预测特权特征。
- **学生策略 (Student Policy)**：输入包括世界模型预测的特征、本体感知历史和参考运动。
- **目标**：三者同步更新。通过蒸馏损失让学生策略模仿教师的动作。

### 第二阶段：策略微调与对齐 (Fine-tuning & Alignment)
- **教师冻结**：教师策略的 Actor 部分被冻结，仅 Critic 部分继续提供价值信号（Advantage）。
- **世界模型对齐**：通过指数移动平均 (EMA) 更新，继续精细化对特权特征的预测。
- **学生进化**：学生策略使用教师 Critic 提供的信号进行强化学习更新。
- **结果**：学生策略学会了在没有真特权信息的情况下，利用世界模型的预测能力完成任务。

## 主要技术路线

| 模块 | 角色 | 训练方法 |
|------|-----|---------|
| **教师策略** | 专家提供者 | 使用特权信息进行标准 RL 训练 |
| **世界模型** | 桥梁/感知层 | 监督学习：利用本体观测预测特权特征 |
| **学生策略** | 部署模型 | 模仿教师动作 + 世界模型预测特征输入 |
| **训练范式** | 两阶段蒸馏 | 从“联合训练”过渡到“固定教师、强化学生” |

## 技术特色

- **特权特征预测**：世界模型不只是预测图像，而是预测对任务关键的物理特征（如“物体现在是否倾斜”）。
- **解耦设计**：感知预测（世界模型）与动作执行（学生 Actor）相互独立又紧密配合。
- **真机鲁棒性**：由于在训练中模拟了预测误差，学生策略在真机部署时对感知噪声具有极强的免疫力。

## 典型应用场景

- **物体搬运**：在不知道物体准确重量的情况下，通过世界模型感知负重并调整姿态。
- **协作任务**：预测合作伙伴的意图或外力变化。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [HAIC 技术报告相关 PDF](../../sources/papers/motion_control_projects.md)

## 关联页面

- [Model-Based RL](./model-based-rl.md) — 世界模型在 HAIC 中作为特征提取器。
- [Privileged Training (特权信息训练)](../concepts/privileged-training.md)
- [Sim2Real](../concepts/sim2real.md) — 教师-学生蒸馏是解决 Sim2Real 差距的标准模式。
