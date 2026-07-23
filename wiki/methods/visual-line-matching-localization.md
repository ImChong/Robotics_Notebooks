---
type: method
tags: [localization, computer-vision, soccer, geometry, humanoid, robocup]
status: complete
updated: 2026-07-23
related:
  - ./soccer-field-line-detection.md
  - ./visual-line-ekf-fusion.md
  - ../concepts/perception-coordinate-postprocessing.md
  - ../tasks/humanoid-soccer.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "空间视觉定位—线匹配：将观测到的场地线/交点与已知场地模型线特征匹配，求解场地上的机器人位姿，是 RoboCup 视觉定位主步骤。"
---

# 线匹配视觉定位

## 一句话定义

**线匹配视觉定位**把图像中提取的 **场地线或交点** 与已知场地图纸上的线特征建立对应，从而估计机器人在场地坐标系中的位姿——课程第 7.2 节。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Data Association | Data Association | 观测线与模型线的对应 |
| PnP | Perspective-n-Point | 点特征位姿求解对照 |
| Homography | Homography | 平面场地单应估计 |
| FOV | Field of View | 可见线数量影响可观测性 |
| Map | Field Model | 规则尺寸的场地先验 |

## 为什么重要

- 足球场有强先验结构，线匹配比纯视觉里程计更抗纹理缺失。
- 输出作为 [EKF 融合](./visual-line-ekf-fusion.md) 的观测更新。
- 像素到场地的变换链见 [感知后处理与坐标变换](../concepts/perception-coordinate-postprocessing.md)。

## 主要技术路线

| 路线 | 观测量 | 求解 |
|------|--------|------|
| 交点–模型匹配 | L/T/X 关键点 | 对应搜索 + 最小二乘 |
| 线–线匹配 | 无限线/线段 | 点到线距离残差 |
| 单应 / 平面 PnP | 平面点集 | Homography 分解位姿 |
| 多假设保留 | 对称场地模糊 | 交 [EKF](./visual-line-ekf-fusion.md) / 粒子滤波 |

## 核心原理

1. 检测线/交点（见 [场地线检测](./soccer-field-line-detection.md)）。
2. 坐标变换到便于匹配的帧（见 [后处理](../concepts/perception-coordinate-postprocessing.md)）。
3. 搜索对应（距离门控、拓扑约束、Ransac）。
4. 最小化重投影/场地平面误差得位姿假设。

## 工程实践

- 至少同时看到 **足够约束的线组合**（如一条线+交点 vs 多线）。
- 模糊对应时保留多假设，交给 EKF/粒子滤波。

## 局限与风险

- 对称场地导致全局定位模糊（左右禁区等）——需运动先验或唯一特征（球门颜色/方向）。
- 遮挡与线断裂导致关联失败。

## 关联页面

- [线特征 EKF 融合](./visual-line-ekf-fusion.md)
- [Humanoid Soccer](../tasks/humanoid-soccer.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- RoboCup 视觉定位开源队代码中的 field-line matcher 实现（各队仓库）
