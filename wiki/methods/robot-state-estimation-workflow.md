---
type: method
tags: [robotics, localization, estimation]
status: draft
updated: 2026-11-26
summary: "融合机器人传感器和运动信息，估计位置、姿态、速度及其不确定性。"
sources:
  - ../../sources/blogs/wechat_shenlan_slam_second_spring_2026-09-02.md
related:
  - ../concepts/state-estimation.md
---

# 机器人状态估计与定位

## 一句话定义

融合机器人传感器和运动信息，估计位置、姿态、速度及其不确定性。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| SLAM | Simultaneous Localization and Mapping | 同时定位与建图 |

## 关联概念

[相关概念](../concepts/state-estimation.md) 描述该方法背景。

## 主要技术路线

1. **观测与标定**：明确 IMU、编码器、相机或激光雷达的坐标系、时间戳及噪声特性。
2. **估计与建图**：根据任务选择滤波、优化或学习方法，区分局部里程计、全局定位和 SLAM 地图。
3. **验证与退化**：测量轨迹误差、漂移、延迟及遮挡或传感器失效后的行为，并核对真实参考轨迹。

[传感器接口](robot-sensor-integration.md) 负责数据接入；[雷达与里程计融合](lidar-odometry-fusion.md) 是特定传感器组合的实例。

## 局限与后续核查

本页提供方法导航，具体配置、结果和适用条件应回到项目及实验资料核对。

## 参考来源

- [相关资料归档](../../sources/blogs/wechat_shenlan_slam_second_spring_2026-09-02.md)
