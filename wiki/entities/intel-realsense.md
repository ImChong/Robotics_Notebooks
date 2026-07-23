---
type: entity
tags: [sensor, depth-camera, realsense, perception, rgb-d, humanoid]
status: complete
updated: 2026-07-23
related:
  - ../methods/object-detection.md
  - ../methods/soccer-field-line-detection.md
  - ../concepts/soccer-field-simulation.md
  - ../entities/unitree-g1.md
  - ../entities/humanoid-system-curriculum.md
  - ../tasks/humanoid-soccer.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "Intel RealSense 深度相机族（D435/D455 等）：消费级 RGB-D，广泛用于人形头部/腕部感知、导航高度图与足球视觉；课程第 6.1 节传感器入口。"
---

# Intel RealSense 深度相机

## 一句话定义

**Intel RealSense** 是一族消费级 **RGB-D 深度相机**（主动红外立体等方案），为机器人提供对齐的彩色与深度图，是人形课程感知章与大量 G1 论文真机的默认视觉传感器之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB-D | RGB + Depth | 彩色与深度对齐帧 |
| D435 / D455 | RealSense Depth Camera models | 常见机载型号 |
| FOV | Field of View | 视场角，影响场地线可见范围 |
| SDK | Software Development Kit | `librealsense` 驱动与工具 |
| IR | Infrared | 主动投影辅助立体匹配 |

## 为什么重要

- 课程第 6 章全部视觉作业以 RealSense 为输入；YOLO 检测与后续线匹配都假设 **已知内参与深度可选**。
- 相对激光成本低、语义友好；相对纯 RGB 多了尺度。

## 核心原理

- 主动立体：红外投影 + 左右 IR 相机算视差 → 深度；RGB 传感器外参对齐。
- 输出：`color`、`depth`、有时 `imu`；ROS 驱动发布光流与点云。

## 工程实践

- 标定：厂内参 + 手眼/头部位姿外参；足球任务对 **俯仰角与场线可见性** 敏感。
- 户外强光/反光地板深度会空洞，检测应允许纯 RGB 回退。
- 与 [目标检测](../methods/object-detection.md)、[场地线检测](../methods/soccer-field-line-detection.md) 组成感知前端。

## 局限与风险

- 阳光、透明/黑色物体深度失效常见。
- 产品线变更时型号停产，选型需看当前 Intel/代理供货。

## 关联页面

- [足球场仿真](../concepts/soccer-field-simulation.md)
- [人形系统课程策展](./humanoid-system-curriculum.md)
- [Unitree G1](./unitree-g1.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- Intel RealSense SDK 2.0 文档：<https://www.intelrealsense.com/>
