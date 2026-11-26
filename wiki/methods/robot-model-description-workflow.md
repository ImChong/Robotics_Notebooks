---
type: method
tags: [robot, urdf, simulation]
status: draft
updated: 2026-11-26
summary: "用结构、关节和物理属性描述机器人，并把模型接入软件与仿真工具。"
sources:
  - ../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md
related:
  - ../concepts/urdf-robot-description.md
---

# 机器人模型描述与资产接入

## 一句话定义

用结构、关节和物理属性描述机器人，并把模型接入软件与仿真工具。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

[相关概念](../concepts/urdf-robot-description.md) 介绍该方法的背景。

## 主要技术路线

1. **结构与坐标**：定义连杆、关节、坐标系、运动范围和装配关系，检查参数单位。
2. **几何与动力学**：分别维护显示、碰撞、质量及惯量数据；URDF、MJCF 和 USD 的能力与转换方式并不相同。
3. **加载和校验**：在目标工具中检查关节顺序、碰撞、坐标系与动力学行为，并注明实际模型与许可证版本。

[机器人软件集成](robot-software-integration.md) 进一步处理驱动、通信和部署接口。

## 局限与后续核查

本页用于跨项目导航；具体接口、功能与许可应以原始项目资料和实际环境为准。

## 参考来源

- [资料归档](../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md)
