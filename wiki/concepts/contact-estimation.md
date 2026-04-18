---
title: Contact Estimation（接触估计）
type: concept
status: complete
created: 2026-04-14
updated: 2026-04-18
summary: 在无直接力传感器时，从关节力矩/速度信号推断足端或末端执行器的接触状态，是步态规划和 WBC 的重要感知前提。
---

# Contact Estimation（接触估计）

## 是什么

Contact Estimation 是指在机器人运动过程中，**实时判断哪个足/末端执行器处于接触状态（与地面或物体接触）**，并尽可能估计接触力的大小和方向。

这是一个感知问题：从可测量量（关节角度、电机电流、IMU）推断不可直接测量量（接触状态、接触力）。

---

## 为什么重要

| 下游模块 | 对接触信息的依赖 |
|---------|--------------|
| WBC / TSID | 需要知道当前接触脚集合，决定约束矩阵 |
| 质心动力学 | 接触力 → 质心加速度，需要接触集合和力 |
| MPC 步态规划 | 相位检测，判断摆动/支撑切换时刻 |
| Sim2Real | 仿真接触 vs 真实接触不一致，影响策略泛化 |
| 跌倒检测 | 意外接触（手腕撞地）需快速检测 |

---

## 常用方法

### 1. 基于力/力矩传感器（Force/Torque Sensor, FT Sensor）
- 足端安装六维 F/T 传感器，直接测量接触力
- **优点**：最精确；**缺点**：昂贵（每只脚 2~5k USD）、增加质量、线缆脆弱

### 2. 基于电机电流/力矩（Torque-Based）
- 从关节力矩估计关节处广义力，通过雅可比反推末端力
  ```
  τ_contact = τ_measured - τ_gravity - τ_dynamics
  f_tip = (J^T)^+ · τ_contact
  ```
- SEA（系列弹性驱动器）机器人：弹性元件形变直接测力
- **缺点**：动力学模型误差影响精度；高频噪声

### 3. 接触状态分类器（Binary Contact Detection）
- 输入：关节力矩残差、足端加速度、支撑相预测
- 输出：0（离地）/ 1（接触）二分类
- 常用：阈值法、HMM（隐马尔可夫模型）、简单神经网络

### 4. 基于 RL 策略的隐式估计
- 现代端到端 RL 策略往往**隐式学习接触状态**，无需显式接触估计模块
- 策略直接从 IMU + 关节状态推断下一步动作
- 缺点：不可解释，WBC 框架无法直接使用

### 5. Kinematic Contact Estimation
- 检查足端运动学速度 + 地形约束：足端速度接近零 → 可能接触
- 结合 IMU 的 Leg Odometry 估计：在 MIT Mini Cheetah、ANYmal 上有应用

---

## 接触力估计（Contact Force Estimation）

在已知接触状态后，进一步估计接触力：

**静态系统（低速）：**
```
J^T · f = τ_gravity_compensation - τ_actuator
→ f = (J^T)^+ · (τ_gravity_compensation - τ_actuator)
```

**动态系统（扩展卡尔曼滤波）：**
- 状态量：`[q, dq, f_contact]`
- 观测量：`τ_joint, a_IMU`
- EKF 在线估计接触力，输出给 WBC

---

## Sim2Real 的接触估计问题

仿真中的接触（硬性约束 + 完美地形）与现实接触（软接触、地形不平）之间存在差异：

- 仿真足端完美接触，真实有滑动/弹跳
- 传感器噪声不匹配 → 估计器在真实机器人上漂移

**缓解方法：**
- 在仿真中加入接触噪声（随机接触延迟、力传感器噪声）
- 使用历史观测（观测窗口）增强对抗噪声能力

---

## 工程实现参考

| 平台 | 接触估计方法 | 采样率 |
|-----|------------|--------|
| MIT Mini Cheetah | 力矩 + 运动学，阈值分类 | 500 Hz |
| ANYmal | SEA 弹性测力 + EKF | 400 Hz |
| Spot | 未公开，推测力矩+IMU | 1000 Hz |
| 人形（G1/H1） | 足端 F/T 传感器为主 | 500 Hz |

---

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — ingest 档案（MuJoCo 接触模型 / CBF 接触一致性）
- Bledt et al., *Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains* (ICRA 2018) — MIT Cheetah 接触估计
- Bloesch et al., *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU* (RSS 2013) — 腿式机器人状态与接触估计
- Camurri et al., *Probabilistic Contact Estimation and Impact Detection for State Estimation of Quadruped Robots* (RA-L 2017) — 概率接触检测
- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — HumanX 说明了接触图模仿奖励与仅靠本体感知历史进行外力估计的做法

---

## 关联页面

- [Whole-Body Control](./whole-body-control.md) — WBC 需要接触状态集合作为输入约束
- [TSID](./tsid.md) — TSID 求解前必须确定接触集合
- [Centroidal Dynamics](./centroidal-dynamics.md) — 接触力是质心加速度的来源
- [Locomotion](../tasks/locomotion.md) — 步态切换依赖实时接触检测
- [Sim2Real](./sim2real.md) — 接触模型差异是 sim2real gap 的重要来源
- [Balance Recovery](../tasks/balance-recovery.md) — 意外接触的快速检测对扰动恢复至关重要
