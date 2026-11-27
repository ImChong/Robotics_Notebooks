# Agent-Driven Autonomous Reinforcement Learning Research: Iterative Policy Improvement for Quadruped Locomotion

- **作者：** Nimesh Khandelwal, Shakti S. Gupta
- **机构：** 印度理工学院坎普尔分校机械工程系（Department of Mechanical Engineering, Indian Institute of Technology Kanpur, Kanpur, India）
- **版本 / 载体：** arXiv:2603.27416v1 [cs.RO]（2026-03-28 提交）
- **原文：** <https://arxiv.org/html/2603.27416>（摘要页：<https://arxiv.org/abs/2603.27416>）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，论文及 arXiv 元数据未列出本研究 DHAV1 Isaac Lab 案例研究实验归档仓库或独立项目页；文中引用的 `DeepRobotics/rl_training`、`DeepRobotics/rl_sar`、`karpathy/autoresearch` 与 `isaac-sim/IsaacLab` 均为外部参考实现或底层仿真框架）
- **核对范围：** 已通读 arXiv:2603.27416v1 HTML 全文（Abstract、Sections I–VIII、Table I–IV、Fig. 1–9 图注与正文分析、References [1]–[16]）；因未公布本研究独立官方代码仓库，未做源码静态核对或本地运行
- **通读：** 已完成

## 原文核对笔记

- **研究定位与人机分工：** 本文是一份聚焦四足机器人运动控制强化学习（RL）的实证案例研究（case study），并非全自动自启动科研系统。人类研究监督者提供高层方向指令（如同时调优 Basic PPO 与 HIM、参考开源粗糙地形配置），运行于 OpenCode 编码智能体环境中的 Claude 智能体承担大部分执行闭环：读取代码、诊断失败模式、修改奖励与地形配置、在 4 块 NVIDIA RTX PRO 4500 Blackwell（每块 32 GB 显存）GPU 上调度与监控训练（通常 4096 并行环境，诊断时 2048 或 512）、终止卡死任务并提出下一波实验。
- **五阶段波次研究闭环：** 共组织 14 个波次（Wave 1–14）、70 余组实验（`exp001`–`exp073`），围绕 DHAV1 12 自由度（12-DoF）四足机器人在 Isaac Lab / Isaac Sim 中的粗糙地形速度跟踪任务展开，每波按「Observation → Hypothesis → Intervention → Execution → Triage」推进。
- **关键自主决策与工程发现（Table I / II / VI-B）：**
  1. **PhysX 死锁根因隔离（Wave 5–6）：** 将环境数减半仅延迟失败而不消除故障，从而排除纯 batch size / buffer 假说；通过对照地形家族，将反复出现的 PhysX 死锁定位到包含 box 与 stair-like 原语的地形组合，切换为受 DeepRobotics（DR）参考实现启发的平滑粗糙地形混合（`random_rough`、`hf_pyramid_slope`、`hf_pyramid_slope_inv`）后首次实现稳定长程训练。
  2. **参考奖励移植与仿真启动修复（Wave 7）：** 从公开参考库（`DeepRobotics/rl_training` / `rl_sar`）移植 4 项奖励：对角步态奖励（diagonal gait reward）、腾空时间方差正则（air-time variance regularization）、关节镜像对称（joint-mirror symmetry）与无指令触地塑形（feet-contact-without-command shaping），并修复 Isaac Sim 启动阶段由 eager import 引起的 `omni`/`pxr` 引导故障。
  3. **策略族分流（Wave 3–8）：** 尽管修复了 HIM（Hybrid Internal Model 分支）的 terminal-observation 处理并测试了架构与跟踪权重调整，HIM 在粗糙地形配置上持续停留在 `terrain=0.0`（如 `exp029`、`exp036` 靠原地站立获得高存活率，但 `vel_err > 1.4`、`reward = 3.1`），智能体遂将主要搜索算力转向 Basic PPO，保留 HIM 作为负对照。
  4. **奖励尺度归一化陷阱与单变量调优（Wave 8–14）：** 总奖励随跟踪权重增大而虚高（如 `exp066` 权重 `8.0/4.0` 使 reward 升至 `174.1`，但 `vel_err` 反而恶化至 `0.282`），论文提出以权重无关的平面速度跟踪误差 `vel_err` 作为公平指标。Wave 11（`exp058`）首次实现该波次 100%（4/4）完成 2000 轮迭代无 PhysX 挂起，`vel_err=0.266`；Wave 12 最优配置 `exp063`（`track=7.0/3.5`、`gait=0.5`、`base_height=-10.0`、`flat_orientation=-10.0`、`airtime=-6.0`）在 2000 轮迭代达到 `reward=153.4`、`vel_err=0.263`、`timeout=97%`（注：`exp058` 报告 `97.7%`，`exp063` 摘要与正文中报告 `97%` / `>97%`），并在 3 块不同 GPU 上独立复现 5 次（含 `exp063`、`exp064`、`exp072`）；Wave 13–14 验证了该配置附近存在局部最优，进一步增大惩罚或加入力矩加倍/关节镜像在当前组合下引发不稳定或性能退化。
- **对 wiki 的映射：** [Agent-Driven Autonomous Reinforcement Learning Research（paper-agentic-2603-27416）](../../wiki/entities/paper-agentic-2603-27416.md)
