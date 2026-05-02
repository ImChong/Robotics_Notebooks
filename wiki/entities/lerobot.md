---
type: entity
title: LeRobot (Hugging Face)
tags: [framework, robot-learning, open-source, dataset]
summary: "LeRobot 是 Hugging Face 开发的具身智能全栈框架，旨在将 Transformers 生态迁移到机器人领域，支持高效数据采集与策略训练。"
updated: 2026-05-01
---

# LeRobot (Hugging Face)

**LeRobot** 是由 Hugging Face 开发并维护的一个**具身智能全栈框架**。它旨在将自然语言处理（NLP）领域的成熟生态（如 `transformers` 库和模型 Hub）迁移到机器人领域，提供从数据采集、策略训练到实物部署的一站式工具。

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

## 参考来源
- [Xbotics-Embodied-Guide](../../sources/repos/xbotics-embodied-guide.md)
- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)
