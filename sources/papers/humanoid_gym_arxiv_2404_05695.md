# Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer（arXiv:2404.05695）

> 来源归档（ingest）

- **标题：** Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer
- **类型：** paper / humanoid locomotion / sim2real / RL 训练框架
- **arXiv：** <https://arxiv.org/abs/2404.05695>（PDF：<https://arxiv.org/pdf/2404.05695>）
- **项目页：** <https://sites.google.com/view/humanoid-gym/>
- **代码：** <https://github.com/roboterax/humanoid-gym>
- **作者：** Xinyang Gu*, Yen-Jen Wang*, Jianyu Chen（RobotEra / 上海期智研究院 / 清华大学）
- **入库日期：** 2026-06-09
- **一句话说明：** 基于 **Isaac Gym + legged_gym 范式** 的人形专用 RL 框架，用 **步态相位奖励 + 非对称 AC 特权训练 + 域随机** 在 **XBot-S / XBot-L** 上实现 **零样本 sim2real**；内置 **Isaac Gym → MuJoCo sim2sim** 校验管线。

## 摘要级要点

- **定位：** 面向人形 **locomotion** 的易用 RL 框架，强调 **零样本仿真到真机**；底层继承 ETH **legged_gym / rsl_rl** 的 `LeggedRobot` 实现。
- **算法：** **PPO** + **Asymmetric Actor-Critic**；训练用完整状态 $s$ 与特权观测，部署在 **POMDP** 下仅用本体观测 $o$。
- **控制接口：** 策略输出 **目标关节位置** → **PD 控制器**（策略 **100 Hz**，内环 PD **1000 Hz**）。
- **步态设计：** **步态周期相位** $C_T$、正弦参考运动、**周期性支撑掩码** $I_p(t)$（DS/SS 四相）；奖励含 **接触模式对齐** $\phi(I_p - I_d)$。
- **观测（单帧 47 维，堆叠 15 帧）：** 时钟 $\sin/\cos$、速度指令、关节 $q,\dot{q}$、基座角速度/欧拉角、上一步动作；特权侧含摩擦、质量、基座线速度、推力/力矩扰动、跟踪差、支撑掩码、足端接触等（单帧 73 维，堆叠 3 帧）。
- **奖励四项：** (1) 速度跟踪 (2) 步态/接触模式 (3) 姿态/高度/关节正则 (4) 能量与动作平滑惩罚；$z,\gamma,\beta$ 速度指令刻意置零以稳走。
- **域随机：** 关节位/速、角速度、欧拉角加性高斯噪声；摩擦、电机强度、载荷、系统延迟等（见论文 Table III）。
- **训练规模（论文）：** 8192 并行环境、episode 2400 步、batch $8192\times24$、lr $10^{-5}$、$\gamma=0.994$。
- **Sim2Sim：** **MuJoCo** 经腿摆正弦与相图校准后动力学 **更接近真机** 而非 Isaac Gym；提供平地与崎岖地形验证。
- **真机验证：** **RobotEra XBot-S（1.2 m）** 与 **XBot-L（1.65 m）** 零样本行走；论文 Fig.1 完整管线示意。
- **后续工作（仓库 README）：** **Denoising World Model Learning（RSS 2024 Best Paper Finalist）**、感知 locomotion 与灵巧手操作（Coming Soon）。

## 对 wiki 的映射

- 沉淀实体页：`wiki/entities/humanoid-gym.md`
- 交叉更新：`wiki/entities/legged-gym.md`、`wiki/concepts/sim2real.md`、`wiki/concepts/domain-randomization.md`、`wiki/tasks/humanoid-locomotion.md`、`references/repos/rl-frameworks.md`
- 姊妹 fork：`sources/repos/humanoid-gym-modified.md`（Pandaman + Gazebo sim2sim）
