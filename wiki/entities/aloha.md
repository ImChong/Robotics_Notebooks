---
type: entity
title: ALOHA (双臂遥操作硬件)
tags: [hardware, teleoperation, dual-arm, dataset-collection]
---

# ALOHA (双臂遥操作硬件)

**ALOHA** (A Low-cost Open-source Hardware System for Bimanual Teleoperation) 是由 Google DeepMind (Tony Zhao 等人) 开发的一套低成本、开源的双臂遥操作硬件系统。它是目前机器人学习领域，尤其是双臂精细操作（Fine Manipulation）中获取人类演示数据的**事实标准**。

## 为什么重要？

在 [[imitation-learning]] 中，演示数据的质量决定了策略的上限。ALOHA 的贡献在于：
- **民主化硬件**：通过使用低成本的电机（ViperX 机械臂），将双臂遥操作系统的成本降低到了数万美元量级。
- **开源生态**：提供了从机械设计图纸到控制代码的全套开源方案。
- **高频交互**：配合 [[action-chunking]] (ACT) 算法，ALOHA 证明了即使是低成本硬件也能完成拉拉链、炒菜等极高难度的动作。

## 核心组成

- **Master 机械臂**：操作员手持的小型机械臂，用于输入指令。
- **Follower 机械臂**：实际执行任务的机器人手臂（通常是 ViperX 300 6DOF）。
- **相机阵列**：通常配备 4 个高清摄像头，提供多视角视觉输入。

## 与其他系统的关系

- **算法搭档**：它是 [[action-chunking]] (ACT) 算法的官方硬件底座。
- **仿真对应**：[[robotwin]] 2.0 等平台常提供 ALOHA 的仿真版本以进行数据增强。
- **进阶版本**：后续发展出了 Mobile ALOHA (具备移动底盘版本) 和更加紧凑的方案。

## 参考来源
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md)
- [ALOHA Project Page](https://tonyzhaozh.github.io/aloha/)
