# 感知与定位（Perception & Localization）

> Ingest 日期：2026-04-15
> 主题：视觉惯性里程计、状态估计的感知角度、传感器融合

---

## 核心论文

### Forster et al. (2017) — SVO: Semi-Direct Visual Odometry
- **核心贡献**：将稀疏直接法与特征点结合，在低算力平台上实现高速 VO
- **关键洞见**：不依赖稠密光流，每帧处理时间 < 2ms；适合腿式机器人实时定位

### Qin et al. (2018) — VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator
- **核心贡献**：单目相机 + IMU 紧耦合非线性优化；滑窗边缘化保证实时性
- **关键洞见**：IMU 预积分消除积分误差；回环检测提供全局一致性；是腿式机器人主流定位方案之一

### Bloesch et al. (2013) — Robust Visual Inertial Odometry Using a Direct EKF-Based Approach
- **核心贡献**：扩展卡尔曼滤波 (EKF) 直接融合图像像素强度与 IMU；无需显式特征提取
- **关键洞见**：对光照变化鲁棒；适合腿式机器人在非结构化环境中定位

### Hartley et al. (2020) — Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation
- **核心贡献**：将接触点信息融入 InEKF（不变扩展卡尔曼滤波）；直接估计基座 pose + 速度
- **关键洞见**：腿式机器人专用状态估计；接触辅助 EKF 减少 IMU 积分误差；是 MIT Cheetah / ANYmal 等平台的标准方案

### Wisth et al. (2022) — VILENS: Visual, Inertial, Lidar, and Leg Odometry for All-Terrain Legged Robots
- **核心贡献**：四模态（视觉 + 惯性 + 激光雷达 + 腿部运动学）融合里程计
- **关键洞见**：多模态冗余在接触失效或遮挡时保持定位；是腿式机器人野外部署的代表工作

---

## Wiki 映射

| 论文 / 概念 | 对应 wiki 页面 |
|-----------|--------------|
| 视觉惯性里程计（VIO） | `wiki/concepts/sensor-fusion.md` |
| InEKF / EKF 状态估计 | `wiki/concepts/state-estimation.md` |
| 接触辅助状态估计 | `wiki/concepts/state-estimation.md` |
| 腿式里程计 | `wiki/concepts/locomotion.md` |
| 传感器融合框架 | `wiki/concepts/sensor-fusion.md` |

---

## 关键结论

1. **VIO 是腿式机器人定位主流**：VINS-Mono / OKVIS / ROVIO 等 VIO 方案在计算效率与精度之间取得平衡，是无 GPS 室内/室外环境的标准选择。
2. **腿式 EKF 的优势**：将接触点约束融入卡尔曼滤波可显著减少漂移；MIT Cheetah 和 Boston Dynamics Atlas 均使用类似方法。
3. **多模态融合是趋势**：单模态在极端环境（光线不足、激光遮挡）下容易失效；VILENS 等多模态方案提升了全地形鲁棒性。
4. **感知-控制耦合**：精确的实时状态估计（位置、速度、脚部接触状态）是 MPC 和 WBC 正常工作的前提。

---

## 参考来源
- SVO: Forster et al., ICRA 2014 / IEEE TRO 2017
- VINS-Mono: Qin et al., IEEE TRO 2018
- Contact-Aided InEKF: Hartley et al., IJRR 2020
- VILENS: Wisth et al., IEEE RA-L 2022
