---
type: concept
tags: [sensor-fusion, perception, localization, vio, ekf, state-estimation]
updated: 2026-06-24
related:
  - ./state-estimation.md
  - ../entities/paper-ultra-fusion-multi-sensor-slam.md
  - ./contact-estimation.md
  - ./floating-base-dynamics.md
  - ../methods/model-predictive-control.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/perception_localization.md
  - ../../sources/papers/state_estimation.md
  - ../../sources/papers/ultra_fusion_arxiv_2606_21223.md
summary: "Sensor Fusion 通过融合 IMU、编码器、视觉等多源信息提升状态估计鲁棒性。"
---

# 传感器融合（Sensor Fusion）

## 一句话定义

**传感器融合**：将来自多个传感器（IMU、摄像头、激光雷达、腿部运动学）的测量值在概率框架下统一融合，估计机器人的位姿、速度和接触状态，为上层控制（MPC / WBC）提供实时、精确的状态输入。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| VIO | Visual-Inertial Odometry | 视觉-惯性里程计，融合相机与 IMU 估计位姿 |
| ANYmal | ANYbotics Quadruped | ANYbotics 的四足机器人研究平台 |
| SOTA | State of the Art | 当前最优水平 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

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

### 退化感知融合（ITS / 导航 SLAM）

移动平台与 ITS 场景中，**弱光照、LiDAR 几何退化、轮速打滑、GNSS 拒止** 会使部分模态残差不可靠；固定权重融合易引入偏置。[Ultra-Fusion](../entities/paper-ultra-fusion-multi-sensor-slam.md)（arXiv:2606.21223）在 **统一滑窗因子图** 内对 LiDAR / 视觉 / IMU / 轮速 / GNSS 做 **因子级可靠性调度**，并配合 **在线时空标定**，在 M3DGR 等基准上对 60+ SLAM 系统做退化与标定扰动评测——与腿式 **InEKF/VIO** 侧重不同，更面向 **轮式/腿式/UAV 多配置导航栈** 的鲁棒定位。

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
- [Ultra-Fusion（多传感器 SLAM 论文实体）](../entities/paper-ultra-fusion-multi-sensor-slam.md) — 退化感知紧耦合 LVIO/LVWIO 与大规模基准
- [状态估计（State Estimation）](./state-estimation.md)
- [接触估计（Contact Estimation）](./contact-estimation.md)
- [浮动基动力学](./floating-base-dynamics.md)
- [MPC](../methods/model-predictive-control.md)
- [全身运动控制（WBC）](./whole-body-control.md)
- [Locomotion 任务](../tasks/locomotion.md)

## 参考来源
- [kalman_filter_ekf_primary_refs.md](../../sources/papers/kalman_filter_ekf_primary_refs.md) — KF / EKF 经典论文与教材索引
- [perception_localization.md](../../sources/papers/perception_localization.md)
- [state_estimation.md](../../sources/papers/state_estimation.md)
