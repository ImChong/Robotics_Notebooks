---
type: method
tags: [control, pid, classical-control, joint-control, quadruped]
status: complete
updated: 2026-11-07
original:
  - ../../sources/books/modern_robotics_ch11_robot_control_reading.md
related:
  - ../concepts/bode-plots.md
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../overview/robot-control-paradigm-classical-linear-feedback.md
  - ./model-predictive-control.md
  - ./reinforcement-learning.md
  - ../comparisons/mpc-vs-rl.md
  - ../entities/quadruped-control-curriculum.md
  - ../concepts/sim2real.md
  - ../concepts/gravity-compensation.md
  - ../concepts/micromouse.md
  - ../entities/ukmarsbot.md
sources:
  - ../../sources/books/modern_robotics_ch11_robot_control_reading.md
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
  - ../../sources/repos/ukmarsbot.md
  - ../../sources/sites/micromouseonline-com.md
summary: "PID 是关节级经典反馈控制：比例-积分-微分组合跟踪期望位置/速度；四足演进路线中作为底层执行层，复杂地形 loco 常由 RL 取代高层。"
---

# PID Control（比例-积分-微分控制）

PID 控制：根据误差 $e$ 及其积分、微分，输出控制量 $u$ 的经典 **线性反馈** 方法，仍是机器人 **关节伺服、姿态稳定** 的默认底层。

## 一句话定义

> 用 **现在误差（P）、历史累积（I）、变化趋势（D）** 三部分加权，把输出拉回设定值。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PID | Proportional–Integral–Derivative | 比例-积分-微分控制 |
| PD | Proportional–Derivative | 足式 loco 常用（省略 I 防 windup） |
| MPC | Model Predictive Control | 预测优化，PID 之后的模型驱动升级 |
| RL | Reinforcement Learning | 数据驱动高层，常仍用 PD 执行 |
| FF | Feedforward | 重力/摩擦前馈与 PID 并联 |
| SDK | Software Development Kit | 实机底层常暴露 PD 增益接口 |

## 为什么重要

几乎所有机器人关节伺服的最底层都是 PID / PD：它不需要完整动力学模型，参数少、可解释，实机上还能用误差响应指标（稳态误差、超调、2% 调节时间）直接验收。更复杂的控制（计算力矩、MPC、RL）通常仍把输出交给底层 PD 去执行，所以看懂它的误差动力学是理解后续一切的前提。

## 核心原理

**控制律（Modern Robotics §11.4.1）：** $\tau=K_p\theta_e+K_i\!\int\!\theta_e\,dt+K_d\dot\theta_e$，$\theta_e=\theta_d-\theta$。$K_p$ 像虚拟弹簧，$K_d$ 像虚拟阻尼，$K_i$ 消稳态误差。

**水平面单关节 PD 的误差动力学：** 设 $M\ddot\theta+b\dot\theta=\tau$，设定点控制下 $M\ddot\theta_e+(b+K_d)\dot\theta_e+K_p\theta_e=0$，即标准二阶形 $\ddot\theta_e+2\zeta\omega_n\dot\theta_e+\omega_n^2\theta_e=0$：

$$
\omega_n=\sqrt{K_p/M},\qquad \zeta=\frac{b+K_d}{2\sqrt{K_pM}}
$$

$\zeta=1$ 临界阻尼（无超调、最快）；原文算例 $M=0.5$ kg·m²、$b=0.1$ N·m·s/rad 时，$K_p=2.205$、$K_d=2$ 刚好 $\zeta=1$。2% 调节时间约 $4/(\zeta\omega_n)$。

**重力使 PD 有稳态误差：** 竖直面时静平衡满足 $K_p\theta_e=mgr\cos\theta$，只能靠增大 $K_p$（受饱和、颤振、结构柔性、采样率限制）减小；加 $K_i>0$ 后误差动力学变三阶，稳定要求

$$
K_d>-b,\quad K_p>0,\quad \frac{(b+K_d)K_p}{M}>K_i>0
$$

即 $K_i$ 既有下界（才有用）也有上界（再大就失稳）；推荐顺序是先按瞬态选 $K_p,K_d$，再取"足够大又不破坏稳定"的 $K_i$。**许多机器人控制器干脆取 $K_i=0$**，并用积分抗饱和（限制积分项幅度）。积分不能消除沿任意轨迹的跟踪误差。

**速度输入的伺服（电机速度模式，原文 §11.3）：** P 控制 $\dot\theta=K_p\theta_e$ 给一阶误差动力学（时间常数 $1/K_p$，调节时间约 $4/K_p$），对恒速参考有稳态误差 $c/K_p$；PI 变二阶，$\omega_n=\sqrt{K_i}$，$\zeta=K_p/(2\sqrt{K_i})$，临界阻尼 $K_i=K_p^2/4$；作者偏好的前馈 + PI 律 $\dot\theta=\dot\theta_d+K_p\theta_e+K_i\!\int\!\theta_e$。

## 在四足控制演进中的位置

课程 Ch1 叙事：

```
PID（单关节伺服）
  → MPC（质心/足端轨迹优化，需模型）
    → RL（复杂地形端到端，PD 仍作执行层）
```

- PID/PD：适合 **已知轨迹跟踪**、桌面级调试、站立平衡初版
- 局限：多关节耦合、接触切换、非结构化地形下 **手工调参难扩展**（MPC 在碎石地也会失效，从而引出 RL）

## 离散形式（位置 PD）

$$
\tau = K_p (q^* - q) + K_d (\dot{q}^* - \dot{q}) + \tau_{\text{ff}}
$$

四足 RL 部署典型栈：**RL 输出 $q^*, \dot{q}^*$ 或 $\tau$** → PD 转力矩 → 电机。

## 工程实践：整定要点（衔接 Sim2Real）

- 增益过高：噪声放大、振荡、过热
- 增益过低：跟踪慢、抗扰差、跌倒
- 课程 Ch6 强调 **实机 PD 整定与安全协议** 与摩擦补偿、DR 并列
- 数字实现（Modern Robotics 伪代码）：速度用相邻采样位置差分再低通；误差积分 `eint += e*dt`；`tau = Kp*e + Kd*edot + Ki*eint`；控制器采样一般数百到数千 Hz，再高收益有限
- 先 $K_p$ 后 $K_d$ 取临界阻尼，再视稳态误差决定是否开 $K_i$；多关节可对角增益 $k_pI$，质量矩阵近似对角（高减速比、无重力）时分散式 PID 即可，重力显著时需重力补偿或计算力矩
- 把已知前馈（重力、摩擦、计划加速度）并联进来，PID 只管残差；完整做法见 [Computed Torque Control](./computed-torque-control.md)


## 频域整定：Bode 图

对线性化关节或电机回路，可用 Bode 图观察目标频段的增益、相位、共振和时延，再评估调整 PID/PD 增益后的跟踪与稳定裕度。裕度应针对实际开环环路传递函数计算；最终还要通过阶跃、扰动和安全的实机负载测试验证。详见[Bode 图概念页](../concepts/bode-plots.md)。

## 主要技术路线

### 1. 独立关节 PID

每关节独立 $K_p, K_i, K_d$，适合 **单轴标定、桌面臂**；足式多关节耦合时增益互相牵制。

### 2. 并联前馈（重力 / 摩擦）

$$
\tau = K_p e + K_d \dot{e} + \tau_{\text{gravity}}(q) + \hat{\tau}_f(\dot{q})
$$

- 重力项常由 [RNEA](../formalizations/articulated-body-algorithms.md) 计算，算法选型见 [Gravity Compensation](../concepts/gravity-compensation.md)
- 摩擦项见 [Joint Friction Models](../concepts/joint-friction-models.md)、[Friction Compensation](../concepts/friction-compensation.md)

### 3. RL + PD 执行层（四足主流）

RL 输出目标位置/力矩，**PD 仍作底层伺服**；复杂地形 loco 由 [PPO](./ppo.md) 等承担，PID 不消失而是 **退居执行层**（见 [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)）。

### 4. 向 MPC / RL 演进

当单关节 PID 无法处理接触切换与非凸地形，升级到 [MPC](./model-predictive-control.md)（模型预测）或 [RL](./reinforcement-learning.md)（数据驱动）；选型见 [MPC vs RL](../comparisons/mpc-vs-rl.md)。

## 局限与常见误区

- 在四足复杂地形上纯 PID 做 loco：缺少接触调度与步态，难以与 RL/MPC 路线竞争。
- 积分项盲目开启：饱和 windup，足式更常用 **PD + 前馈**；Modern Robotics 还给出明确的稳定上界 $K_i<(b+K_d)K_p/M$。
- 以为增大 $K_p$ 就能消除重力误差：增益受执行器饱和、颤振、结构振动与有限伺服频率限制，应改用重力补偿（见 [Gravity Compensation](../concepts/gravity-compensation.md)）。
- 忽视执行器本身：关节"力矩"并不理想——电流环、减速器摩擦 / 回差、SEA 弹簧都会改变实际带宽（原文 §11.8）。

## 关联页面

- [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md)
- [经典线性反馈控制（体系①）](../overview/robot-control-paradigm-classical-linear-feedback.md)
- [Model Predictive Control](./model-predictive-control.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [MPC vs RL](../comparisons/mpc-vs-rl.md)
- [Friction Compensation](../concepts/friction-compensation.md)
- [Gravity Compensation](../concepts/gravity-compensation.md) — PD + $g(q)$ / 迭代学习前馈
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)
- [Micromouse](../concepts/micromouse.md) — 轮速环 / 侧墙跟随的嵌入式竞赛场景
- [UKMARSBOT](../entities/ukmarsbot.md) — 差速入门平台上的速度环练习

## 推荐继续阅读

- Lynch & Park, *Modern Robotics* §11.2–§11.4（误差动力学、PID、计算力矩）：<http://modernrobotics.org>
- Åström & Murray, *Feedback Systems*（本次未能取得全文，作为延伸阅读保留）
- [Humanoid RL Cookbook](../queries/humanoid-rl-cookbook.md) — PD 增益与部署

## 参考来源

- **原文通读归档：** [Modern Robotics 第 11 章](../../sources/books/modern_robotics_ch11_robot_control_reading.md)（2026-11-07）
- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝八大控制体系：经典线性反馈代表算法
- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch1 控制演进
- [sources/repos/ukmarsbot.md](../../sources/repos/ukmarsbot.md)
- [sources/sites/micromouseonline-com.md](../../sources/sites/micromouseonline-com.md)
