---
type: entity
tags: [humanoid, urdf, model, kinematics, roboparty]
status: complete
updated: 2026-04-25
related:
  - ./roboto-origin.md
  - ./atom01-hardware.md
  - ./atom01-train.md
sources:
  - ../../sources/repos/atom01_description.md
summary: "Atom01_description 提供 Atom01 的 URDF 与网格模型资源，是仿真、控制与可视化共享的统一模型基线。"
---

# Atom01 Description

**atom01_description** 是 Atom01 的机器人描述仓库，主要提供 URDF、网格和模型配置，用于连接硬件实体与仿真/控制软件。

## 为什么重要

- 训练、仿真、部署共用同一模型描述可降低系统不一致。
- 为动力学建模、碰撞模型和可视化提供统一基准。
- 是 Sim2Sim/Sim2Real 问题定位的重要锚点。

## 核心结构/机制

- **URDF 模型**：关节拓扑、链接参数、惯性与约束。
- **网格资源**：可视化与碰撞几何。
- **模型配置**：供训练与部署管线读取。

## 常见误区或局限

- 误区：URDF 正确就等于真机可控。实际上还受摩擦、延迟、驱动器参数误差影响。
- 局限：仅描述静态模型结构，无法单独覆盖固件时序与执行器非线性。

## 参考来源

- [sources/repos/atom01_description.md](../../sources/repos/atom01_description.md)
- [Roboparty/atom01_description](https://github.com/Roboparty/atom01_description)

## 关联页面

- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [Atom01 Hardware](./atom01-hardware.md)
- [Atom01 Train](./atom01-train.md)

## 推荐继续阅读

- [sources/urdf.md](../../sources/urdf.md)
- [Atom01 Deploy](./atom01-deploy.md)
