# VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator

> 来源归档：原作者论文出版入口；针对 3D 视觉与 SLAM 课程目录补齐缺失的主论文，非课程发布者文章。

- **标题：** VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator
- **类型：** paper
- **原论文：** https://ieeexplore.ieee.org/document/8421746
- **官方实现：** https://github.com/HKUST-Aerial-Robotics/VINS-Mono
- **会议/期刊：** IEEE Transactions on Robotics 2018
- **入库日期：** 2026-11-08
- **一句话说明：** 以相机观测与 IMU 预积分为约束，完成视觉惯性初始化与紧耦合非线性优化状态估计。
- **核对范围：** 已读完 arXiv:1708.03852v1 方法 §III–VIII、IMU preintegration/covariance、视觉初始化/重力与尺度、滑窗边缘化、DBoW2 回环、4DOF pose graph、§IX EuRoC/MAV/移动端实验及 §X 结论，核对作者官方 ROS 代码入口；未重跑论文实验。
- **通读：** 已完成

## 原文逐节审阅结果

- §IV：KLT 光流追踪、RANSAC 和 IMU bias-aware 预积分，位姿/速度/陀螺与加计零偏 covariance 同步传播。
- §V：五点法/多视 SfM 得到 **up-to-scale** 轨迹，用 IMU 预积分估计陀螺零偏、gravity direction、velocity、metric scale。高动态激励使 scale 可观，静止长期不必然自动收敛。
- §VI：滑窗同时优化位姿、速度、相机-IMU 外参、逆深度与 IMU bias；目标是 marginalized prior、preintegration inertial residual、unit sphere 视觉重投影（Eq.22）。非关键帧边缘化保留惯性约束。
- §VII–VIII：额外 BRIEF+DBoW2 候选、2D-2D fundamental/RANSAC 与 3D-2D PnP 拒真值错配，再重定位；gravity roll/pitch 可观使全局只优化 x/y/z/yaw 4 DoF，而非任意 6 DoF。
- §IX 实验：EuRoC 20fps 单目+200Hz IMU（**原始数据有双目，论文只用左相机**）；700m 室内外跑：纯 VIO 终点漂移比例约 **0.88%**，回环终点误差 `[-0.032,0.09,-0.07]m`（无全程 GT，不得冒充 ATE）；5.62km 校园 1h34m 的感知率图表显示 feature 25Hz、滑窗 10Hz（i7-4790）。实际设备和代码版本影响结果。
- 论文局限：相机/IMU 同步与外参标定、初始激励不足、长时间特征丢失与错误回环导致失败；新版仓库增加 2017 temporal calibration/rolling-shutter，但旧论文 v1 未给出新版数值保证。

## 方法摘要（以出版方摘要为依据）

以相机观测与 IMU 预积分为约束，完成视觉惯性初始化与紧耦合非线性优化状态估计。

## 源码与论文身份

- 原论文：[VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator](https://ieeexplore.ieee.org/document/8421746)
- 代码：[https://github.com/HKUST-Aerial-Robotics/VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
- 总清单：[视觉与三维研究一手来源](../sites/vision3d_research_primary_sources_2026_11_08.md)

## 复现与局限核查

- [x] 原论文全文、EuRoC 与 MAV/大尺度示范均核对；已标明非 GT 轨迹的误差定义。
- [x] 已核对官方 launch 和各 ROS 节点主回调，作者身份明确。
- [x] [VINS-Mono 独立节点](../../wiki/entities/paper-vins-mono.md) 包含两类 Mermaid。
