---
type: concept
tags: [sensor-fusion, perception, localization, vio, ekf, state-estimation]
related:
  - ./state-estimation.md
  - ./contact-estimation.md
  - ./floating-base-dynamics.md
  - ../methods/model-predictive-control.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/perception_localization.md
  - ../../sources/papers/state_estimation.md
summary: "Sensor Fusion 通过融合 IMU、编码器、视觉等多源信息提升状态估计鲁棒性。"
---

# 传感器融合（Sensor Fusion）

## 一句话定义

**传感器融合**：将来自多个传感器（IMU、摄像头、激光雷达、腿部运动学）的测量值在概率框架下统一融合，估计机器人的位姿、速度和接触状态，为上层控制（MPC / WBC）提供实时、精确的状态输入。

## 为什么重要

> 精确的实时状态估计是 MPC 和 WBC 正常工作的**前提条件**。状态估计误差会直接导致步态不稳、接触检测失败、轨迹跟踪偏差。

传感器融合解决的核心问题：IMU 积分快速漂移；单目视觉缺乏尺度；接触状态噪声大。只有多模态融合才能在全地形下保持稳健估计。

---

## 主流方法

### 视觉惯性里程计（VIO）

将摄像头与 IMU 紧耦合，是无 GPS 室内/室外定位的标准方案。

| 方案 | 类型 | 特点 |
|------|------|------|
| **VINS-Mono** | 单目 + IMU，非线性优化 | 滑窗边缘化，实时，腿式主流 |
| **SVO** | 半直接法 VO | < 2ms/帧，适合低算力平台 |
| **ROVIO** | 单目 + IMU，EKF | 直接使用像素强度，光照鲁棒 |

**IMU 预积分**：消除帧间 IMU 积分误差，保证优化效率，是 VIO 的核心技巧。

---

### 接触辅助 InEKF（腿式机器人专用）

**Contact-Aided Invariant Extended Kalman Filter（InEKF）**：
- 将脚部接触点约束融入卡尔曼滤波的观测模型
- 直接估计基座 pose（位置 + 姿态）+ 线速度
- 接触点提供稳定的"零速度更新"，显著减少 IMU 漂移
- MIT Cheetah、ANYmal、Boston Dynamics Atlas 的标准状态估计方案

**为何用不变 EKF（InEKF）而非标准 EKF**：
- 标准 EKF 在 SE(3) 上线性化误差大
- InEKF 利用李群结构，误差传播与初始状态无关，收敛更快

---

### 多模态融合（全地形鲁棒）

**VILENS**（视觉 + 惯性 + 激光雷达 + 腿部运动学）：
- 四模态冗余设计：单模态失效时其余模态补偿
- 激光雷达补偿视觉弱纹理区域失效
- 代表了腿式机器人野外部署的 SOTA 方向

---

## 在控制栈中的位置

```
传感器（IMU / 摄像头 / 激光雷达 / 关节编码器）
          ↓
  传感器融合 / 状态估计
  （InEKF / VIO / VILENS）
          ↓
  基座 pose + 速度 + 接触状态
          ↓
  MPC（质心轨迹规划） → WBC（全身力矩控制）
```

**延迟敏感性**：MPC 典型运行频率 100–500 Hz；状态估计必须在同等或更高频率输出，否则成为瓶颈。

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 漂移快 | IMU 积分误差累积 | 视觉回环检测 / 接触约束更新 |
| 空中腿干扰估计 | 非接触腿运动学噪声 | 接触检测 + 加权融合 |
| 光线不足失效 | 视觉特征丢失 | 激光雷达补偿 / 热成像 |
| 地形突变跳变 | 接触冲击产生高频噪声 | 低通滤波 + 异常检测 |

---

## 关联页面
- [状态估计（State Estimation）](./state-estimation.md)
- [接触估计（Contact Estimation）](./contact-estimation.md)
- [浮动基动力学](./floating-base-dynamics.md)
- [MPC](../methods/model-predictive-control.md)
- [全身运动控制（WBC）](./whole-body-control.md)
- [Locomotion 任务](../tasks/locomotion.md)

## 参考来源
- [perception_localization.md](../../sources/papers/perception_localization.md)
- [state_estimation.md](../../sources/papers/state_estimation.md)
