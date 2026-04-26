---
type: formalization
tags: [state-estimation, kalman-filter, ekf, filtering, locomotion]
status: complete
related:
  - ../concepts/state-estimation.md
  - ../concepts/sensor-fusion.md
  - ../concepts/contact-estimation.md
sources:
  - ../../sources/papers/state_estimation.md
  - ../../sources/papers/perception_localization.md
summary: "Extended Kalman Filter (EKF)"
updated: 2026-04-25
---

# Extended Kalman Filter (EKF)

**扩展卡尔曼滤波（EKF）**：将标准卡尔曼滤波推广到非线性系统的经典状态估计方法，通过每步线性化（一阶 Taylor 展开）在非线性系统上近似应用 Kalman 递推公式。

## 一句话定义

> EKF 是"把卡尔曼滤波硬套到非线性系统上"的工程化方案——每步在当前估计点附近线性化非线性模型，然后按标准 KF 递推。不精确但实用，是足式机器人状态估计里二十年来最常用的方法之一。

## 为什么重要

人形/足式机器人的状态估计（base pose、velocity、contact）涉及：
- IMU 的非线性旋转积分
- 足端运动学（sin/cos 非线性）
- 接触状态切换

这些都超出标准 KF 的线性假设。EKF 是最直接的非线性扩展，被广泛用于：
- base state estimation（基座姿态与速度估计）
- IMU + 编码器 + 足端接触融合
- 里程计与 VIO 的初始化

## 标准 Kalman Filter 回顾

线性系统：

$$x_{k+1} = A x_k + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)$$

$$z_k = C x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)$$

各矩阵的物理含义：

- $A$：状态转移矩阵，刻画无控输入时状态如何随时间演化
- $B$：控制输入矩阵，将控制量 $u_k$ 映射到状态空间
- $C$：观测矩阵，将状态投影到传感器测量空间
- $Q$：过程噪声协方差（$w_k$ 的协方差），表示模型不确定度
- $R$：观测噪声协方差（$v_k$ 的协方差），表示传感器噪声大小
- $P_{k|k}$：状态估计的后验协方差，反映当前估计置信度
- $K_k$：卡尔曼增益，权衡“相信预测”与“相信观测”
- $I$：与状态维度匹配的单位矩阵

**预测步**：

$$\hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_k$$

$$P_{k|k-1} = A P_{k-1|k-1} A^T + Q$$

**更新步**：

$$K_k = P_{k|k-1} C^T (C P_{k|k-1} C^T + R)^{-1}$$

$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - C \hat{x}_{k|k-1})$$

$$P_{k|k} = (I - K_k C) P_{k|k-1}$$

## EKF：非线性扩展

非线性系统：

$$x_{k+1} = f(x_k, u_k) + w_k$$

$$z_k = h(x_k) + v_k$$

**关键思路**：在当前估计点 $\hat{x}$ 处做一阶 Taylor 展开，得到雅可比矩阵代替线性矩阵 $A$、$C$：

$$F_k = \left. \frac{\partial f}{\partial x} \right|_{\hat{x}_{k-1|k-1}}, \quad H_k = \left. \frac{\partial h}{\partial x} \right|_{\hat{x}_{k|k-1}}$$

**EKF 预测步**：

$$\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}, u_k)$$

$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q$$

**EKF 更新步**：

$$K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1}$$

$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - h(\hat{x}_{k|k-1}))$$

$$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$

## 在足式机器人中的应用

### 典型状态向量

$$x = [p_b, v_b, \theta_b, p_{f_1}, \ldots, p_{f_n}]$$

其中：$p_b$ 基座位置，$v_b$ 基座速度，$\theta_b$ 基座姿态（四元数或 RPY），$p_{f_i}$ 足端位置。

### 传感器融合

| 传感器 | 提供信息 | 用于 |
|--------|----------|------|
| IMU（加速度+陀螺仪） | 基座加速度、角速度 | 预测步（$f$） |
| 编码器（关节角度） | 运动学约束 | 更新步（$h$）|
| 接触传感器/压力 | 足端是否着地 | 更新步条件激活 |
| 相机/激光（可选） | 全局位姿 | 更新步（VIO）|

## InEKF（不变 EKF）

EKF 在旋转相关状态上有一致性问题（observability 不一致）。**InEKF（Invariant EKF）** 利用 Lie 群结构，把状态定义在 $SE(3)$ 等流形上，得到：

- 更好的观测性（理论一致性更强）
- 在旋转估计上比普通 EKF 更鲁棒

代表工作：Hartley et al. *Contact-Aided Invariant EKF*，是当前足式机器人状态估计的主流算法之一。

## EKF 的局限

| 问题 | 描述 |
|------|------|
| 线性化误差 | 严重非线性区域一阶近似不够准 |
| 高斯假设 | 状态分布强非高斯时失效（如接触切换瞬间）|
| 观测性问题 | 某些状态组合不可观测，EKF 协方差可能发散 |
| 接触切换 | 支撑脚换腿时需要处理状态不连续 |

替代方案：
- **UKF**（Unscented KF）：用 sigma points 处理非线性，不做线性化
- **Particle Filter**：非参数化，适合强非高斯，但计算量大
- **优化方法**（Factor Graph / MAP）：用滑窗优化代替滤波，精度更高

## 关联页面

- [State Estimation](../concepts/state-estimation.md) — EKF 是状态估计的核心算法，在该页有完整的使用场景描述
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md) — 浮动基机器人状态估计的动力学背景
- [Sim2Real](../concepts/sim2real.md) — 状态估计精度直接影响 sim2real 效果
- [LQR / iLQR](./lqr.md) — LQR/EKF 是最优控制中的"最优估计+最优控制"经典对

## 参考来源

- Kalman, *A New Approach to Linear Filtering and Prediction Problems* (1960) — KF 原始论文
- Hartley et al., *Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation* (2020) — 足式机器人 InEKF 代表
- Barrau & Bonnabel, *The Invariant Extended Kalman Filter as a Stable Observer* (2017) — InEKF 理论基础

## 推荐继续阅读

- [State Estimation](../concepts/state-estimation.md)
- Hartley et al., *Contact-Aided Invariant Extended Kalman Filtering*（InEKF 实现参考）

## 一句话记忆

> EKF 是"每步线性化的卡尔曼滤波"——适用于非线性机器人系统，是足式机器人基座状态估计的二十年经典基础算法，InEKF 是其 Lie 群改进版。
