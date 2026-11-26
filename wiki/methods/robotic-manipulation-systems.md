---
type: method
tags: [manipulation, robotics, hardware]
status: draft
updated: 2026-11-26
summary: "将机械臂、末端执行器、感知和控制接口组合为可执行的物体操作系统。"
sources:
  - ../../sources/datasets/robot-manipulation-primary-sources-2026-11-07.md
related:
  - ../concepts/whole-body-control.md
---

# 机器人操作系统与末端执行器

## 一句话定义

将机械臂、末端执行器、感知和控制接口组合为可执行的物体操作系统。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

[相关概念](../concepts/whole-body-control.md) 描述本方法的背景。

## 主要技术路线

1. **机械结构与末端**：明确自由度、工作空间、负载、抓手和安装接口；手臂与夹爪性能应按完整系统评估。
2. **感知与控制**：依据任务选择位置、速度、力或柔顺控制接口，核对相机标定、坐标系、通信与安全限位。
3. **任务验证**：以目标物体、初始状态和成功判据评估可达性、稳定性与恢复能力；机械规格不等同于任务成功率。

[机器人操作任务](../tasks/manipulation.md) 从任务角度组织抓取、搬运和装配；本页关注系统如何集成。

## 局限与后续核查

本页提供跨项目导航，具体硬件性能、实现、版本和评测以相应资料为准。

## 参考来源

- [相关资料归档](../../sources/datasets/robot-manipulation-primary-sources-2026-11-07.md)
