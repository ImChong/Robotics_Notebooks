---
type: method
tags: [navigation, robotics, planning]
status: draft
updated: 2026-11-26
summary: "将目标、环境感知、路径规划和运动控制连接为可验证的机器人导航流程。"
sources:
  - ../../sources/repos/navigation2.md
related:
  - ../concepts/hierarchical-quadruped-navigation-stack.md
---

# 机器人导航与路径执行

## 一句话定义

将目标、环境感知、路径规划和运动控制连接为可验证的机器人导航流程。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| SLAM | Simultaneous Localization and Mapping | 同时定位与建图 |

## 关联概念

[相关概念](../concepts/hierarchical-quadruped-navigation-stack.md) 描述该方法背景。

## 主要技术路线

1. **目标与环境**：明确导航目标、地图、坐标系、机器人尺寸与可通行区域；未知环境需独立处理探索。
2. **规划与执行**：结合全局路径、局部避障和运动控制，并随传感器与状态估计更新计划。
3. **评测与边界**：报告到达率、碰撞、延迟、任务环境和人工接管；仿真效果不等于真实环境表现。

[A*](a-star.md) 与 [动态窗口法](dwa.md) 分别是全局搜索与局部规划的具体示例，并非所有系统均采用这两种方法。

## 局限与后续核查

本页提供方法导航，具体配置、结果和适用条件应回到项目及实验资料核对。

## 参考来源

- [相关资料归档](../../sources/repos/navigation2.md)
