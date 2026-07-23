---
type: concept
tags: [perception, coordinate-transform, calibration, robotics, soccer]
status: complete
updated: 2026-07-23
related:
  - ../formalizations/3d-coordinate-transforms-vision-robotics.md
  - ../formalizations/homogeneous-coordinates-transform.md
  - ../methods/soccer-field-line-detection.md
  - ../methods/visual-line-matching-localization.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "感知后处理与坐标变换：把检测框/线特征从像素系变换到相机、机器人 base 与场地世界系，并做滤波与拓扑校验，是足球视觉闭环的胶水层。"
---

# 感知后处理与坐标变换

## 一句话定义

**感知后处理与坐标变换**把检测器输出的 **像素量** 经相机模型与外参链变换为 **机器人/场地坐标系下的几何量**，并做置信度过滤与拓扑校验——课程第 7.1 节。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TF | Transform Frame | ROS 坐标树常用简称 |
| \(T^a_b\) | Transform a←b | 齐次变换约定需统一 |
| Intrinsics | Camera Intrinsics | \(K\) 矩阵，像素↔射线 |
| Extrinsics | Camera Extrinsics | 相机到 base/头的位姿 |
| NMS | Non-Maximum Suppression | 检测后处理之一 |

## 为什么重要

- 检测对了但坐标系错，EKF 会系统性偏场。
- 统一后处理层可复用到导航目标点、踢球瞄准等模块。

## 核心原理

典型链：

`pixel → camera ray/depth → camera frame → base_link → odom/map/field`

要点：

1. 深度缺失时用 **场地平面假设** 求交。
2. 时间戳对齐（相机 vs 关节/IMU）。
3. 拒绝低置信或几何不可能测量。

详见 [三维坐标变换](../formalizations/3d-coordinate-transforms-vision-robotics.md)。

## 工程实践

- 用 TF 树可视化 camera→base；标定板或手工测外参。
- 日志记录：原始检测 + 变换后场地点，便于回放。

## 局限与风险

- 平面假设在机器人俯仰大时误差放大。
- 左右相机/深度对齐误差会进定位协方差。

## 关联页面

- [线匹配视觉定位](../methods/visual-line-matching-localization.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- [齐次坐标与变换](../formalizations/homogeneous-coordinates-transform.md)
