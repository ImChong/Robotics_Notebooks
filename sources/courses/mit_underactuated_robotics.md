# MIT Underactuated Robotics（Russ Tedrake）

> 来源归档（course / open textbook）

- **标题：** Underactuated Robotics — Algorithms for Walking, Running, Swimming, Flying, and Manipulation
- **类型：** course
- **主讲：** Russ Tedrake（MIT）
- **主站：** <https://underactuated.mit.edu/index.html>
- **入库日期：** 2026-09-23
- **版本注记：** 站点标注 Spring 2024 学期持续更新；Preface 强调 **HTML 在线版为主版本**，PDF 次之。

## 为什么值得保留

- 机器人 **欠驱动动力学 + 计算优化 + 学习** 的一条龙公开教材；与 CMU 16-745 Optimal Control 互补（见 [`cmu_optimal_control_16_745_2025_youtube.md`](./cmu_optimal_control_16_745_2025_youtube.md)）。
- 全书示例绑定 **Drake**；与本库 [`drake.md`](../../wiki/entities/drake.md)、轨迹优化、MPC、足式/操作任务页强相关。
- Preface 明确教学哲学：**螺旋式** 按问题引入技术（pendulum → cart-pole/acrobot → walking → …），而非按章节线性读完。

## 全书结构（四大部分）

| 部分 | 章节 | 主题 |
|------|------|------|
| **Model Systems** | 1–6 | 欠驱动定义、摆/Acrobot/Cart-Pole/四旋翼、步行跑步简化模型、ZMP/CoP、随机性模型系统 |
| **Nonlinear Planning and Control** | 7–17 | DP/HJB、LQR、Lyapunov/SOS、TrajOpt/MPC、Policy Search、采样规划、鲁棒/随机控制、反馈运动规划、输出反馈、极限环、接触混杂系统 |
| **Estimation and Learning** | 18–21 | 系统辨识、状态估计（KF 等）、无模型策略搜索、模仿学习（含 Diffusion Policy） |
| **Appendix** | A–E | Drake、优化工具箱、组合优化、几何与接触 |

## 章节 → 本库 wiki 映射（精选）

### Model Systems（Ch 1–6）

| 章 | 要点 | Wiki 映射 |
|----|------|-----------|
| 1 | 全驱动 vs 欠驱动、非完整约束 | [Optimal Control](../../wiki/concepts/optimal-control.md) |
| 2–3 | 单摆、能量整形、PFL、LQR 局部稳定 | [LQR](../../wiki/formalizations/lqr.md)、[LQR/iLQR](../../wiki/methods/lqr-ilqr.md) |
| 4 | Rimless Wheel、Compass Gait、SLIP、极限环 | [Locomotion](../../wiki/tasks/locomotion.md)、[LIP/ZMP](../../wiki/concepts/lip-zmp.md) |
| 5 | ZMP、CoP、Centroidal、Whole-Body Control | [Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 6 | 随机接触、MDP 基础 | [MDP](../../wiki/formalizations/mdp.md) |

### Nonlinear Planning and Control（Ch 7–17）

| 章 | 要点 | Wiki 映射 |
|----|------|-----------|
| 7 | DP、HJB、值迭代 | [Bellman 方程](../../wiki/formalizations/bellman-equation.md) |
| 8 | LQR 族（时变、流形、约束、凸形式） | [LQR](../../wiki/formalizations/lqr.md) |
| 9 | Lyapunov、Barrier、SOS、收缩度量 | [Control Lyapunov Function](../../wiki/formalizations/control-lyapunov-function.md) |
| 10 | TrajOpt、MPC、iLQR/DDP、微分平坦 | [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)、[MPC](../../wiki/methods/model-predictive-control.md) |
| 11–12 | Policy Search、PRM/RRT | [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) |
| 13 | 随机 LQR、鲁棒 MPC、H∞ | [Safe RL](../../wiki/methods/safe-rl.md) |
| 14–15 | 反馈运动规划、LQG、pixels-to-torques | [State Estimation](../../wiki/concepts/state-estimation.md)、[VLA](../../wiki/methods/vla.md) |
| 16–17 | 极限环、接触混杂、contact-implicit TO | [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)、[Manipulation](../../wiki/tasks/manipulation.md) |

### Estimation and Learning（Ch 18–21）

| 章 | 要点 | Wiki 映射 |
|----|------|-----------|
| 18 | 系统辨识、神经网络动力学 | [System Identification](../../wiki/concepts/system-identification.md) |
| 19 | KF、Bayes 滤波、平滑 | [Kalman Filter](../../wiki/formalizations/kalman-filter.md)、[EKF](../../wiki/formalizations/ekf.md) |
| 20 | Policy Gradient、REINFORCE | [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) |
| 21 | Behavior Cloning、Diffusion Policy | [Imitation Learning](../../wiki/methods/imitation-learning.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md) |

## 与 CMU 16-745 的分工

- **Underactuated（Tedrake）：** 欠驱动足式/操作模型系统 → 优化与学习并重的 **螺旋式** 教材；Drake 代码贯穿。
- **CMU Optimal Control（Manchester）：** OCP/LQR/MPC/TrajOpt/DDP 的 **录像 + notebook** 系统课。
- 勿混 YouTube「Optimal Control 2025」playlist 归属 — 见 [`mit_underactuated_kalman_lqr.md`](./mit_underactuated_kalman_lqr.md) 澄清。

## 对 wiki 的映射

- 策展实体：[`wiki/entities/painode-074-mitunderactuatedrobotics.md`](../../wiki/entities/painode-074-mitunderactuatedrobotics.md)
- 站点归档：[`sources/sites/mit-underactuated-robotics.md`](../sites/mit-underactuated-robotics.md)

## 当前提炼状态

- [x] 首页 TOC + Preface/Organization 摘录
- [x] 章节 ↔ wiki 映射表
- [x] Drake / CMU OC 交叉澄清
- [ ] 后续：按学期标注 YouTube playlist 年度差异
