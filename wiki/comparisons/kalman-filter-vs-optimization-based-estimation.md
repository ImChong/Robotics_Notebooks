---
type: comparison
tags: [estimation, perception, math, optimization, filter, sensor-fusion]
status: complete
updated: 2026-04-21
related:
  - ../concepts/state-estimation.md
  - ../formalizations/mdp.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/perception.md
summary: "状态估计方法对比：探讨了经典的卡尔曼滤波（EKF/UKF）与新兴的基于优化的估计（Batch Optimization / VIO）在延迟、精度、鲁棒性及计算资源上的差异。"
---

# Kalman Filter vs. Optimization-based Estimation (状态估计选型)

在机器人（特别是人形和四足机器人）中，实时估计 Base 的位置、速度和姿态是所有算法的基础。目前主要存在两大技术路线：以 **EKF** 为代表的递归滤波派，和以 **滑窗优化 (Sliding Window Optimization)** 为代表的优化派。

## 核心对比

| 维度 | Kalman Filter (如 EKF, InEKF) | Optimization-based (如 VIO, Factor Graph) |
|------|------------------------------|-------------------------------------------|
| **数学本质** | 马尔可夫递归更新 | 最小化非线性代价函数 |
| **时域处理** | 仅保留“当前”状态，丢弃历史 | 同时处理一个时间窗口内的所有观测（滑窗） |
| **非线性处理** | 一阶泰勒展开（线性化），易漂移 | 多次迭代优化，对非线性拟合更准 |
| **计算开销** | 极小，适合嵌入式 MCU | 较大，通常需要高性能 CPU/GPU |
| **延迟/实时性** | ✅ 极低，确定性高 | ❌ 存在迭代耗时，在高频运控中需小心 |
| **鲁棒性** | ❌ 易受奇异点和初值偏差影响 | ✅ 支持 Re-linearization，对传感器异常更鲁棒 |

## 1. 递归滤波派：以 EKF 为核心
- **代表算法**：Invariant EKF (InEKF)。
- **适用场景**：底层 **1kHz 的 IMU + 腿部运动学融合**。由于计算极快，它几乎没有延迟，是 WBC 内部状态估计的标准配置。
- **缺点**：一旦发生严重的传感器跳变（如脚打滑），EKF 很难通过单步更新纠正过来。

## 2. 优化派：以因子图 (Factor Graph) 为核心
- **代表算法**：VINS-Mono, OKVIS, LIO-SAM。
- **适用场景**：**视觉/激光里程计**（Vision/LiDAR Odometry）。这类传感器频率较低（10-30Hz），但信息丰富，通过滑窗优化可以实现厘米级的长距离定位精度。
- **优点**：可以方便地加入闭环检测（Loop Closure）和异步传感器。

## 机器人架构中的融合趋势

现代高性能机器人通常采用**分层估计**架构：
1. **脊髓级 (Fast Path)**：使用 **InEKF** 以 1000Hz 融合 IMU 与关节数据，输出用于平衡控制的极低延迟速度估计。
2. **大脑级 (Slow Path)**：使用 **滑窗优化** 以 30Hz 融合视觉、点云和 IMU，输出用于导航的长程漂移修正量。

## 关联页面
- [State Estimation (状态估计)](../concepts/state-estimation.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [MDP 形式化](../formalizations/mdp.md)

## 参考来源
- [sources/papers/perception.md](../../sources/papers/perception.md)
- Sola, J. *Course on SLAM*.
- Bloesch, M., et al. (2015). *Robust Visual-Inertial Odometry via Inexact Iterated Kalman Filtering*.
