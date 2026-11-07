---
type: method
tags:
  - control
  - model-based
  - manipulation
  - computed-torque
  - dynamics
status: complete
updated: 2026-11-07
original:
  - ../../sources/books/modern_robotics_ch11_robot_control_reading.md
summary: "机械臂经典非线性控制：模型前馈线性化 + PD 反馈消除残差。"
related:
  - ../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md
  - ./inverse-dynamics-control.md
  - ./feedback-linearization-control.md
  - ../entities/pinocchio.md
  - ../concepts/gravity-compensation.md
sources:
  - ../../sources/books/modern_robotics_ch11_robot_control_reading.md
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Computed Torque Control（计算力矩控制，CTC）

CTC：用动力学模型计算前馈力矩抵消非线性耦合，再叠加反馈使闭环近似线性解耦系统。

## 一句话定义

> CTC：用动力学模型计算前馈力矩抵消非线性耦合，再叠加反馈使闭环近似线性解耦系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CTC | Computed Torque Control | 计算力矩控制 |
| FF | Feedforward | 模型前馈 |
| PD | Proportional–Derivative | 常见反馈层 |

## 为什么重要

是理解 WBC / 逆动力学控制的入门枢纽：它把"非线性动力学 + 线性反馈"拆开，使每个关节的误差动力学变成可按极点配置的线性系统。Modern Robotics 注记：计算力矩起源于 1970 年代，其计算复杂度与建模误差问题推动了此后的非线性控制、鲁棒控制、迭代学习控制与自适应控制研究。（此前页面写的"工业协作臂标准中层方案"属经验表述，未在原文核实。）

## 核心原理

$\tau = M(q)(\ddot{q}_d + K_p e + K_d \dot{e}) + C(q,\dot{q})\dot{q} + g(q)$；前馈抵消 $C,g$，反馈处理模型误差。

**推导（Modern Robotics §11.4.1.3、§11.4.2）：** 机器人动力学 $\tau=M(\theta)\ddot\theta+h(\theta,\dot\theta)$，$h$ 含 Coriolis、重力、摩擦等只依赖状态的项。若希望误差 $\theta_e=\theta_d-\theta$ 满足线性误差动力学 $\ddot\theta_e+K_d\dot\theta_e+K_p\theta_e+K_i\!\int\!\theta_e\,dt=0$，则令指令加速度 $\ddot\theta=\ddot\theta_d+K_d\dot\theta_e+K_p\theta_e+K_i\!\int\!\theta_e$，代入**模型** $\{\tilde M,\tilde h\}$：

$$
\tau=\tilde M(\theta)\Big(\ddot\theta_d+K_p\theta_e+K_i\!\int\!\theta_e\,dt+K_d\dot\theta_e\Big)+\tilde h(\theta,\dot\theta)
$$

模型完美时每个关节的误差动力学恰好是上面的线性方程，增益按极点配置选（实践中 $K_i$ 常取 0，$K_p=k_pI$ 等对角）。该控制器等价于"前馈 + 反馈线性化"，也叫**逆动力学控制**。

**与其他方案的比较（原文仿真，$r$ 模型误差 0.08 vs 真值 0.1 m）：** 单独前馈在模型有误差时误差较大；单独反馈要靠高增益；计算力矩的跟踪优于二者，控制能量小于单独反馈。**分散式** PID（每关节独立）只适用于动力学近似解耦的情形——笛卡尔 / 龙门机器人，或高减速比、无重力的机器人（质量矩阵近似对角）；重力显著或质量矩阵非对角时用上述集中式计算力矩。若动力学太复杂、算不动，且期望速度 / 加速度较小，可退化为"PID + 重力补偿" $\tau=K_p\theta_e+K_i\!\int\!\theta_e+K_d\dot\theta_e+\tilde g(\theta)$（见 [Gravity Compensation](../concepts/gravity-compensation.md)）。任务空间版本为 $\tau=J_b^\top(\theta)\big(\tilde\Lambda(\theta)(\dot{\mathcal V}_d+\dots)+\tilde\eta\big)$（操作空间形式，原文 §11.4.3）。

## 工程实践

Pinocchio/RNEA 实时算 $\tau$（Modern Robotics 的 `ComputedTorque` 函数以 $\tilde M\cdot(\ddot\theta_d+K_p e+K_d\dot e+K_i\!\int\!e)+\tilde h$ 的形式实现）；Sim 标定惯性参数；重力项见 [Gravity Compensation](../concepts/gravity-compensation.md)，并与 [Friction Compensation](../concepts/friction-compensation.md) 并联。

## 主要技术路线

### 1. 完整 CTC 前馈 + PD 反馈

文内代表实现路径；详见 [关联概念/形式化](../concepts/friction-compensation.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

参数失配导致残余耦合；高速运动需考虑柔性未建模项。具体地（Modern Robotics）：
- **模型精度与算力：** 完整动力学计算量大，且需要较准的惯量、摩擦模型——这正是计算力矩之后出现鲁棒控制、自适应控制、迭代学习控制的原因（原文 §11.9 简述：鲁棒控制显式保证有界参数不确定下的性能；自适应控制在线估计惯性参数；ILC 利用上一次执行的轨迹误差修正下一次的前馈）。
- **控制器之外的现实：** 柔性 / 振动、齿轮回差、执行器饱和、传感器分辨率不在原文推导范围内；摩擦模型"仍是活跃研究领域"，简单模型只含粘滞摩擦 $b\dot\theta$。
- **执行层：** 假设关节能精确输出所要求的力矩；实际靠电流环（典型 10 kHz 内环 + 1 kHz 力矩外环）、应变片反馈或 SEA 实现，减速器摩擦与回差使力矩存在不确定性。
- **积分项：** $K_i>0$ 会带来稳定性代价，因此常取 0，需要时配抗饱和。

## 关联页面

- [Model-based Nonlinear Dynamics](../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md)
- [Inverse Dynamics Control](./inverse-dynamics-control.md)
- [Pinocchio](../entities/pinocchio.md)
- [Gravity Compensation](../concepts/gravity-compensation.md) — CTC 前馈中的 $g(q)$

## 参考来源

- **原文通读归档：** [Modern Robotics 第 11 章](../../sources/books/modern_robotics_ch11_robot_control_reading.md)（2026-11-07）
- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- Lynch & Park, *Modern Robotics* §11.4（计算力矩 / PID / PD + 重力补偿）：<http://modernrobotics.org>
- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

