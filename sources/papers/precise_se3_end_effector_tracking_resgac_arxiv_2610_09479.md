# Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control（arXiv:2610.09479）

> 来源归档（ingest）

- **标题：** Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control
- **arXiv：** <https://arxiv.org/abs/2610.09479>（v1，2026-10-07）；[HTML](https://arxiv.org/html/2610.09479v1)
- **项目 / 补充页：** [ResGAC](https://resgac.github.io/ResGAC-website/)；[实验详情](https://resgac.github.io/ResGAC-website/details.html)
- **作者 / 机构：** Joohwan Seo、Xiaofeng Guo、Jinkun Cao、Roberto Horowitz、Rocky Duan、Guanya Shi、Koushil Sreenath；UC Berkeley、Amazon Frontier AI & Robotics、CMU
- **平台：** Unitree G1（29 DoF：14 个手臂关节 + 15 个腿/腰关节）
- **代码状态（2026-10-10）：** 未发现公开代码仓库；项目页说明匿名评审期间框架名称与仓库暂不公开。
- **一句话说明：** ResGAC 以几何导纳控制提供可解释的末端位姿跟踪底座，再用残差强化学习补偿动态误差并协调腿/腰，使 G1 在站立、移动基座和行走扰动下保持精确双手跟踪。

## 核心摘录

### 方法
论文针对人形机器人移动或承受全身扰动时的精确笛卡尔末端跟踪。单独的几何导纳控制（GAC）能将 SE(3) 位姿误差映射为手臂关节目标，但不足以处理步行、接触和未建模动力学；端到端 RL 则需要同时学精确几何控制与平衡。

ResGAC 将两者放进同一关节位置目标接口：GAC 产生名义手臂目标，RL 预测有界残差修正手臂目标，并直接输出其余 15 个腿/腰关节目标，学习步态和平衡协调。

### 参考坐标系
论文提出地面附着的 heading frame H0：保留平面位置和偏航，剔除 pelvis 的 roll、pitch、heave。相较将世界目标直接转入随 pelvis 移动的坐标系，H0 可减少躯干运动传入手部参考；它是坐标选择而非消除真实扰动的控制器。

### 实验结果
- **站立轨迹（G1 真机）：** setpoint MAE 为 6.5 mm / 1.37°；pick-and-place 为 6.6 mm / 1.67°。四种留出轨迹上优于 E2E RL、GAC + decoupled RL、SONIC；相对最佳基线位置误差降低约 54–67%，方向误差约 72%。
- **Peg-in-hole：** 25 mm 销插入 35 mm 孔，18/20（90%）；SONIC v1.1 为 10/20（50%）。失败包括目标注册误差与 OptiTrack 遮挡。
- **移动基座世界坐标跟踪：** ResGAC 8.8 mm / 2.69°，E2E RL 30 mm / 13.75°。
- **行走静态手保持：** H0 将 roll 传递比从 0.958 降到 0.790、z MAE 从 22.5 mm 降到 12.6 mm；pitch 传递没有改善（0.954 对 0.964）。

## 原文摘要

> Precise Cartesian end-effector tracking is essential for humanoid robots performing contact-rich tasks while walking or under whole-body disturbances. This work proposes ResGAC, a hybrid framework combining geometric admittance control (GAC) with residual reinforcement learning (RL). GAC provides a structured nominal action for end-effector tracking, while residual RL compensates for dynamic effects and coordinates whole-body motion. The method achieves accurate SE(3) tracking on a Unitree G1 humanoid across standing, moving-base, and walking scenarios, including real-world peg-in-hole insertion.

## Wiki 映射

- 论文实体：wiki/entities/paper-resgac-precise-se3-end-effector-tracking.md
- 任务综述：wiki/tasks/loco-manipulation.md
- 方法综述：wiki/methods/residual-policy-learning.md
