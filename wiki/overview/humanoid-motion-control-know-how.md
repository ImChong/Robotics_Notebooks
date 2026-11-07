---
type: overview
tags: [humanoid, control, know-how, engineering, deployment]
status: complete
updated: 2026-11-07
original:
  - ../../sources/papers/kim_wbic_mpc_arxiv_1909_06586.md
  - ../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md
  - ../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
sources:
  - ../../sources/papers/kim_wbic_mpc_arxiv_1909_06586.md
  - ../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md
  - ../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
  - ../../sources/papers/sim2real.md
  - ../../sources/papers/whole_body_control.md
summary: "人形机器人运动控制工程实践经验汇总：涵盖了从传感器噪声处理、电机热管理到非结构化地形适应的闭坑指南。"
---

# 人形机器人运动控制 Know-How

> **说明**：本文档沉淀了在真实人形机器人研发与调试过程中的硬核工程经验。不同于教科书上的理论，这里的每一条建议往往都对应着一次“炸机”或烧毁电机的惨痛教训。
>
> **证据等级（2026-11-07 起）**：下文每条经验分两类——标有「原文核对」的数字或结论已用上面四份一手论文核对；标有「经验（推测）」的数值（如滤波截止频率、摩擦系数建议值、温升数据）来自维护者的工程经验，**一手论文未给出**，使用前请按自己的机型复核。

## 一句话定义

人形运动控制 Know-How 是一份把「传感器与状态估计 → 全身控制 / 强化学习策略 → 执行器与电气」这条链路上最常见的真机坑位整理成清单的工程备忘，目的是在上机前预判故障，而不是替代教材或论文。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| MPC | Model Predictive Control | 滚动时域优化，WBC 的上层足端反力规划（Kim 2019 中 30–40 Hz） |
| WBIC | Whole-Body Impulse Control | 以反力指令为核心的全身控制，Kim 2019 提出 |
| PD | Proportional–Derivative | 关节级位置 / 阻抗内环，RL 策略输出其 setpoint |

## 为什么重要

- 真机故障常常不出在算法，而出在传感器安装、估计链路、执行器能力与供电这些「教科书不写」的环节。
- 一手论文给出了可核对的量级：多速率控制（MPC 30–40 Hz、WBC 500 Hz、关节 PD 40 kHz）、策略 50 Hz 配 1 kHz PD、执行器带宽与延迟必须建模等，可作为排查时的基准。

## 核心原理：三层栈的一手依据（原文核对）

| 层 | 原文依据 | 要点 |
|----|----------|------|
| 多速率结构 | Kim et al. 2019（Mini-Cheetah） | 状态估计与 WBIC 0.5 kHz，MPC 约 30–40 Hz，关节 PD 40 kHz；MPC 低频规划反力，WBC 高频反馈补足模型与位置控制的缺陷 |
| 摩擦锥 | 同上，式 (11) | 摩擦锥以四棱锥近似：$\lvert f_x\rvert\le\mu f_z,\ \lvert f_y\rvert\le\mu f_z,\ f_z>0$；论文未给 $\mu$ 取值 |
| 执行器 | Kim 2019；Hwangbo 2019 | 指令力矩超过上限时输出被截断（Mini-Cheetah 髋指令 25.5 N·m 对额定 17 N·m）；ANYmal 策略用满 40 N·m 与 12 rad/s；理想执行器模型训出的策略真机上一步都走不了，须用网络或精细模型对齐 |
| 观测 | Hwangbo 2019 | IMU 只能给出姿态的两个自由度（重力方向）；基座高度不可观测，靠腿运动学 + 一维卡尔曼估计，且腾空时不可用；作者推测关节状态历史让策略能检测接触，省去足端力传感器 |
| RL 策略栈 | Radosavovic 2023（Digit） | 策略 50 Hz、关节 PD 1 kHz；动作含 PD setpoint 与腿关节预测增益；闭链膝 / 踝机构在 Isaac Gym 中用高刚度虚拟弹簧近似；随机化覆盖动力学、控制参数、地形与观测噪声 / 延迟 |
| 延迟与异步 | DreamWaQ++ 2026 | 本体 200 Hz、外感知 10 Hz、控制器 50 Hz 异步；本体延迟随机化 0–15 ms，更大会过保守；用 SE(3) 对齐的 K 帧点云记忆代替重建 |

## 工程实践：经验条目（按子系统）

### 1. 传感器与状态估计

#### 1.1 IMU 的“幻觉”
- **高频振动**：足端撞击地面的瞬时振动会通过骨架传导至 IMU。如果固定不牢，加速度计会因共振产生巨大偏置。
- **解决**：使用硅胶减震垫；在软件层使用 200Hz 以上的采样率并配合 Butterworth 低通滤波。*（经验（推测）：一手论文未涉及 IMU 减振与滤波参数。）*
- **原文核对**：Hwangbo 2019 指出 IMU 只能观测姿态两个自由度，用「重力方向单位向量」代替完整姿态作观测；Kim 2019 的状态估计也是在 0.5 kHz 下用卡尔曼滤波融合运动学与加速度。

#### 1.2 接触估计 (Contact Estimation)
- **阈值陷阱**：仅靠足底压力传感器（Force Sensor）判断触地极不靠谱，因为地面的反弹和草地的虚位会产生误报。
- **融合逻辑**：结合关节编码器的速度突变和高度图信息进行多源判定。
- **原文核对**：Hwangbo 2019 推测关节状态历史足以让策略检测接触，因此无需会增加末端重量的足端力传感器；Watanabe 2025（aibo）则因消费级机器人无力传感，使用二值足底开关接触，并发现去掉它后策略学不会走。

### 2. 动力学与控制 (WBC/RL)

#### 2.1 摩擦锥的线性化
- 在 QP 求解器中，摩擦锥通常被线性化为四棱锥。**注意**：摩擦系数 $\mu$ 在实机上通常比仿真里要小（建议取 0.4-0.6 进行保守估计），以防止脚底打滑。*（0.4–0.6 为经验（推测）；Kim 2019 只给出四棱锥不等式，未给 $\mu$ 值。作为对照，RMA 把训练摩擦随机化到 0.05–4.5，DreamWaQ++ 取 0.2–1.25。）*

#### 2.2 奇异点保护
- 当腿部完全伸直（膝关节奇异点）时，雅可比矩阵会退化，导致控制力矩瞬间爆炸。
- **对策**：在控制律中加入虚拟弹簧力，阻止关节进入极限位置前 2-5 度的区域。*（2–5 度为经验（推测）。原文核对：Hwangbo 2019 指出直接输出关节命令的学习策略可在任意构型求值，无需针对奇异点的分支处理，这是 RL 策略相对模型法的优点。）*

### 3. 硬件与电气工程

#### 3.1 电池压降
- 高动态动作（如跳跃）会导致母线电压从 48V 瞬间降至 40V 以下。
- 必须确保逻辑计算板（IPC）拥有独立的稳定供电，否则主控会因欠压重启，导致机器人失去平衡。
- *（48V→40V 为经验（推测），一手论文未给电池压降数据。）*

#### 3.2 热管理
- 连续行走 15 分钟后，膝关节电机通常会达到 70°C 以上。
- **经验**：在 WBC 中加入能效项，减少不必要的对抗出力。
- *（15 分钟 / 70°C 为经验（推测），一手论文未给温升数据。相关的原文事实：Kim 2019 的 Mini-Cheetah 在 3.7 m/s 时髋关节指令力矩远超额定，持续高强度运行的热与功率余量需自行评估。）*

## 局限与风险

- 本页的数值多数是单个团队的工程经验，不能当作规范；标注「经验（推测）」的条目尚无一手论文支撑。
- 一手论文覆盖的平台（Mini-Cheetah、ANYmal、Digit、Go1）与读者的人形平台不同，量级可迁移但数值不可照搬。
- 本页不覆盖电池 / 热管理的论文证据，相关内容请转 [人形机器人电池与热管理指南](../queries/humanoid-battery-thermal-management.md)。

## 关联页面
- [人形机器人 RL 运动控制：身体系统栈视角](./humanoid-rl-motion-control-body-system-stack.md) — 把 42 篇 RL 运动控制论文按 8 层系统栈组织的总框架视图
- [人形机器人电池与热管理指南](../queries/humanoid-battery-thermal-management.md)
- [野外机器人排障指南](../queries/field-robotics-troubleshooting.md) — 应对非结构化地形下的感知与平衡失效
- [Sim2Real 工程 Checklist](../queries/sim2real-checklist.md#快速部署检查)

## 参考来源
- [Kim 2019 WBIC + MPC 论文归档（arXiv:1909.06586，2026-11-07 通读）](../../sources/papers/kim_wbic_mpc_arxiv_1909_06586.md)
- [Hwangbo 2019 执行器网络论文归档（arXiv:1901.08652）](../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md)
- [Radosavovic 2023 Digit 论文归档（arXiv:2303.03381）](../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md)
- [DreamWaQ++ 论文归档（arXiv:2409.19709）](../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md)
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md)
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)

## 推荐继续阅读

- [Highly Dynamic Quadruped Locomotion via WBIC and MPC（arXiv:1909.06586）](https://arxiv.org/abs/1909.06586) — MPC + WBC 多速率架构的一手出处
- [Learning Agile and Dynamic Motor Skills for Legged Robots（arXiv:1901.08652）](https://arxiv.org/abs/1901.08652) — 执行器建模与观测设计的经典案例
- [Real-World Humanoid Locomotion with Reinforcement Learning（arXiv:2303.03381）](https://arxiv.org/abs/2303.03381) — 全尺寸人形 RL 真机部署
