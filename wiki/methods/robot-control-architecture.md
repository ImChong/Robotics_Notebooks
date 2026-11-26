---
type: method
tags: [robotics, control, architecture]
status: draft
updated: 2026-11-26
summary: "组织任务命令、状态估计、运动控制与执行器接口，并明确各层运行边界。"
sources:
  - ../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md
related:
  - ../concepts/whole-body-control.md
---

# 机器人控制系统架构

## 一句话定义

组织任务命令、状态估计、运动控制与执行器接口，并明确各层运行边界。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

[相关概念](../concepts/whole-body-control.md) 描述本方法的背景。

## 主要技术路线

1. **命令与参考**：确定目标位姿、速度或力，以及任务层与运动层之间的接口。
2. **状态与控制**：利用传感器和状态估计闭环计算控制量；位置、力矩与力控制的频率和约束各不相同。
3. **部署与验证**：检查通信、时延、限位、紧急停止和日志；仿真中的稳定性需要独立的真机验证。

[轨迹优化](trajectory-optimization.md) 可生成受约束参考；[Sim2Real 工作流](sim2real-transfer-workflow.md) 关注部署前后的差异。

## 局限与后续核查

本页提供跨项目导航，具体硬件性能、实现、版本和评测以相应资料为准。

## 参考来源

- [相关资料归档](../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md)
