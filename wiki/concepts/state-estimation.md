---
type: concept
tags: [state-estimation, ekf, kalman, imu, contact]
status: complete
updated: 2026-04-20
summary: "State Estimation 负责从传感器中恢复机器人姿态、速度和接触状态，是控制闭环的前提。"
---

# State Estimation

**State Estimation（状态估计）**：根据传感器观测、机器人模型和历史信息，估计机器人当前最可能真实状态的过程。

## 一句话定义

控制器不是直接看见“真实世界”，它看到的是带噪声、带延迟、甚至不完整的传感器数据。

所以状态估计本质上是在回答：

> 机器人现在到底处于什么状态？

比如：
- 基座姿态是多少
- 质心速度是多少
- 足端是否真的接触地面
- IMU 有没有漂
- 关节速度是否可信

## 为什么重要

控制、规划、学习这三件事，前面都要先有状态。

如果状态错了，后面再牛的控制器也会被带沟里。

在人形机器人里，这一点尤其狠，因为：
- 浮动基不可直接测量
- 接触状态会切换
- IMU 会漂
- 编码器有噪声
- 地面接触和足底打滑会让速度估计变差

一句话：

> 没有靠谱的状态估计，MPC、TSID、WBC、RL policy 都是在闭着眼开车。

## State 到底指什么

在机器人里，“状态”通常不只是一个位置。

典型状态包括：
- 关节位置 \( q \)
- 关节速度 \( \dot{q} \)
- 基座位置 / 姿态
- 基座线速度 / 角速度
- 质心位置 / 速度
- 接触状态
- 偏置项（比如 IMU bias）
- 有时还包括外力、摩擦参数等扩展状态

对于固定底座机械臂，状态估计相对简单。

但对人形 / 四足 / 无人机这种 floating-base 系统，最难的通常是：
- base pose
- base velocity
- contact state

## 人形机器人里最难估的几类量

### 1. Base Pose / Base Velocity
因为机身不是固定在地上的，躯干位姿没法像机械臂那样直接从关节角唯一算出来。

你通常需要融合：
- IMU
- 关节编码器
- 足端接触信息
- 有时还要相机 / VIO / LiDAR

### 2. Contact State
脚是不是稳稳踩在地上，直接影响：
- 运动学约束是否成立
- 速度积分会不会漂
- 控制器能不能信任支撑脚

接触状态估错，会让整个状态估计一起歪。

### 3. COM / Momentum
很多高层控制器关心的是：
- 质心位置
- 质心速度
- 整体动量

这些量往往不能直接测，只能通过模型和观测推出来。

## 基本思路：预测 + 校正

状态估计最经典的思路是两步：

### 1. Prediction
用系统模型预测下一时刻状态：

$$
x_{k+1}^- = f(x_k, u_k)
$$

### 2. Update / Correction
拿传感器观测来修正预测：

$$
z_k = h(x_k) + v_k
$$

本质上就是：
- **模型** 告诉你“按理说会变成什么样”
- **传感器** 告诉你“现实看起来像什么样”
- **估计器** 在这两者之间做平衡

## 常见方法

### 1. Complementary Filter
非常工程化、非常常用。

典型场景：
- IMU 陀螺积分给短时姿态
- 加速度计给长期重力方向参考

优点：
- 简单
- 快
- 稳

缺点：
- 表达能力有限
- 不适合复杂多源状态融合

### 2. Kalman Filter（KF）
适合线性高斯系统。

标准线性系统：

$$
x_{k+1} = A x_k + B u_k + w_k
$$
$$
z_k = C x_k + v_k
$$

优点：
- 有闭式递推
- 理论清晰

缺点：
- 线性假设太强

### 3. Extended Kalman Filter（EKF）
机器人里最经典的状态估计器之一。

做法：
- 用非线性模型
- 每步线性化
- 然后按 Kalman 形式更新

常用于：
- IMU + 编码器 + 足端接触融合
- base state estimation
- legged robot state estimation

### 4. Unscented Kalman Filter（UKF）
不直接做一阶线性化，而是用 sigma points 传播不确定性。

优点：
- 有时比 EKF 更稳

缺点：
- 计算更重

### 5. Optimization-based Estimation
把状态估计写成滑窗优化或 MAP 问题。详见 [Kalman Filter vs. Optimization-based Estimation 选型对比](../comparisons/kalman-filter-vs-optimization-based-estimation.md)。

常见于：
- VIO / SLAM（如 [LingBot-Map](../methods/lingbot-map.md)）
- 多传感器融合
- 高精度全局定位

机器人里如果感知复杂，这条路很强，但实时性和工程复杂度也更高。

## 最小代码骨架

这段代码把状态估计最小闭环写清楚：
- 用模型做一步预测
- 再拿传感器观测做一次校正
- 感受“预测 + 更新”比裸传感器读数更稳

```python
x = 0.0          # 估计的位置
v = 1.0          # 已知速度
u_dt = 0.1
z_meas = 0.12    # 传感器观测
alpha = 0.2      # 简化的校正增益

# prediction
x_pred = x + v * u_dt

# correction
x_est = x_pred + alpha * (z_meas - x_pred)
print("predicted:", x_pred)
print("estimated:", x_est)
```

真实机器人里，只是把这里的一维量，换成 base pose / velocity / contact / bias 等高维状态，再把 `alpha` 换成 EKF / UKF / optimization-based estimator 的正式更新律。

## 方法局限性

- **强依赖传感器质量**：IMU 漂、接触误判、编码器噪声都会直接污染估计结果
- **模型错了也会估偏**：预测模型太粗糙时，滤波器只是在“稳定地犯错”
- **多传感器融合很考工程**：时间同步、坐标系标定、延迟补偿往往比公式更难
- **观测 gap 是 sim2real 的硬问题**：仿真里状态干净，真机上观测脏，很多策略就是死在这里

## 人形 / 足式机器人中的典型传感器

### 1. IMU
提供：
- 角速度
- 线加速度

优点：快。
缺点：积分漂。

### 2. Joint Encoders
提供：
- 关节位置
- 有时有速度估计

优点：关节层直接可测。
缺点：对 floating base 无法单独解决。

### 3. Foot Contact Sensors / F/T Sensors
提供：
- 接触是否建立
- 足底力信息

优点：对 gait phase 和接触约束非常关键。
缺点：接触边界时常常不干净。

### 4. Vision / VIO / LiDAR
提供：
- 全局或半全局定位参考
- 外界几何信息

优点：能抑制长时间漂移。
缺点：计算重、延迟高、对环境有依赖。

## 在控制链里的位置

State Estimation 通常位于控制闭环最前面：

```text
传感器数据
    ↓
State Estimator
    ↓
得到 base / joint / contact / CoM / velocity 等状态
    ↓
MPC / Centroidal Planner / TSID / WBC / Policy
    ↓
控制输出
```

所以它不是一个边角模块，而是全控制链的入口。

## 和已有页面的关系

- [System Identification](./system-identification.md)（状态估计依赖机器人动力学模型，SysID 是模型可信度的前提）
- [Floating Base Dynamics](./floating-base-dynamics.md)（floating base 状态估计是 state estimation 最难的部分之一）
- [EKF / InEKF](../formalizations/ekf.md)（EKF 是状态估计的核心滤波算法，见独立 formalization 页）

### 和 TSID 的关系
TSID 要算任务误差、雅可比项、动力学一致解，前提是当前姿态、速度、接触状态得估得靠谱。

见：[TSID](./tsid.md)

### 和 Whole-Body Control 的关系
WBC 想稳住身体、控制摆腿、分配接触力，都依赖状态估计提供的 base 和 contact 信息。

见：[Whole-Body Control](./whole-body-control.md)

### 和 Centroidal Dynamics 的关系
如果质心位置、速度、动量都估不准，那 centroidal planner 也会算偏。

见：[Centroidal Dynamics](./centroidal-dynamics.md)

### 和 Sim2Real 的关系
现实世界里传感器噪声、延迟、偏置都会进来，所以 sim2real 不只是动力学 gap，观测 gap 也很要命。

见：[Sim2Real](./sim2real.md)

## 常见坑

### 1. 把观测当真值
传感器读数不是地面真值，只是 noisy measurement。

### 2. 忽略延迟
控制 1 kHz，视觉 30 Hz，还带几十毫秒延迟，这不处理会很难受。

### 3. 接触判断太脆
脚底轻微滑动、碰撞、软接触都会让 contact estimator 抖。

### 4. 模型过强，观测过弱
如果模型权重太大，估计器会“自我感动”，现实早偏了它还觉得自己对。

### 5. 估计器和控制器目标不一致
控制器关心的是稳定控制，不一定是最漂亮的全局最优定位。所以估计器要为控制服务，不是纯比赛刷精度。

## 为什么它在学习控制里也重要

很多人一提状态估计就只想到传统控制，其实 RL / IL 一样离不开它。

因为策略输入如果来自现实传感器，那么：
- 观测是否对齐
- 速度是否可靠
- 接触状态是否可用
- 延迟和噪声是否建模

都会直接决定 sim2real 效果。

所以很多 deployment-ready policy 成功的关键，不只是 reward 和网络结构，而是 observation pipeline 够不够干净。

## 参考来源

- [sources/papers/state_estimation.md](../../sources/papers/state_estimation.md) — ingest 档案（Bloesch 2013 / Hartley InEKF 2020 / Teng 2021）
- Bloesch et al., *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU* — 足式机器人状态估计经典
- Hartley et al., *Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation* — 接触辅助 EKF 代表
- Barrau, Bonnabel, *The Invariant Extended Kalman Filter as a Stable Observer* — InEKF 理论基础

## 推荐继续阅读

- [传感器融合（Sensor Fusion）](./sensor-fusion.md) — VIO / InEKF / 多模态融合的实现细节
- Bloesch et al., *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU*
- Hartley et al., *Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation*
- Barrau, Bonnabel, *The Invariant Extended Kalman Filter as a Stable Observer*

## 一句话记忆

> State Estimation 解决的是“机器人现在真实处于什么状态”，它是所有控制、规划和 sim2real 闭环的入口，没有它，后面的聪明算法都容易失明。
