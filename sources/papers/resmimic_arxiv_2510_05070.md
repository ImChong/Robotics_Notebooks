# ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning（arXiv:2510.05070）

> 来源归档（ingest）

- **标题：** ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning
- **类型：** paper / humanoid / loco-manipulation / motion-tracking / residual-learning / reinforcement-learning / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2510.05070>
- **arXiv HTML：** <https://arxiv.org/html/2510.05070>
- **PDF：** <https://arxiv.org/pdf/2510.05070>
- **项目页：** <https://resmimic.github.io/>
- **代码：** <https://github.com/amazon-far/ResMimic> — 归档见 [`sources/repos/resmimic.md`](../repos/resmimic.md)
- **机构：** Amazon FAR（Frontier AI & Robotics）；USC；Stanford；UC Berkeley；CMU（Siheng Zhao、Yanjie Ze 实习于 Amazon FAR；† Amazon FAR Co-Lead）
- **硬件：** Unitree G1（29 DoF，1.3 m）
- **仿真：** IsaacGym（训练）；MuJoCo（sim-to-sim 评测）
- **入库日期：** 2026-06-09
- **一句话说明：** **两阶段残差学习**：大规模 **GMT 预训练** 提供人形全身运动先验，**任务残差策略** 在物体参考条件下精修动作，配合 **点云物体跟踪奖励**、**接触奖励** 与 **虚拟力课程**，在 G1 真机实现 **表达力强、全身接触丰富** 的 loco-manipulation。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://resmimic.github.io/> | 真机视频、基线对比、残差动作可视化、消融；归档见 [`sources/sites/resmimic-github-io.md`](../sites/resmimic-github-io.md) |
| 代码仓库 | <https://github.com/amazon-far/ResMimic> | GPU 加速仿真基础设施、sim-to-sim 评测原型、运动数据（论文承诺发布） |
| GMT 基线实现 | TWIST [6] | 运动跟踪奖励与域随机化沿用 TWIST 配方 |
| 重定向 | GMR [39] | 人体 MoCap → 人形参考轨迹 |
| MoCap 数据 | AMASS [8]、OMOMO [9] | GMT 阶段 >15k clips（约 42 h）人类动作 |
| 相近 loco-manip | Dao et al. [10]、Liu et al. [11] | 分阶段 / 轨迹优化参考；ResMimic 强调 **统一框架 + 全身接触** |
| 残差先例 | ASAP [22] | 人形残差补偿 sim–real 动力学差；ResMimic 残差面向 **物体交互** |
| 同机构框架 | [holosoma](https://github.com/amazon-far/holosoma) | Amazon FAR 人形 RL + 重定向；OmniRetarget 数据生成互补 |

## 摘要级要点

- **问题：** GMT 能复现多样人类动作，但 **缺乏物体感知与操作精度**；现有 loco-manipulation 多 **任务定制管线**（分阶段控制器、手工数据），难扩展。
- **洞察：** 平衡、迈步、伸手等 **全身运动跨任务共享**；仅 **细粒度物体交互** 需任务适配 → **预训练 GMT + 轻量残差修正**。
- **Stage I — GMT：** 仅用人类 MoCap（AMASS+OMOMO 等）训练 $\pi_{\mathrm{GMT}}$，输出粗动作 $a^{\mathrm{gmt}}$；奖励 $r^m$ 与 TWIST 同族；**无特权信息** 单阶段 PPO。
- **Stage II — Residual：** 冻结/复用 GMT，训练 $\pi_{\mathrm{Res}}$ 输出 $\Delta a^{\mathrm{res}}$，$a=a^{\mathrm{gmt}}+\Delta a^{\mathrm{res}}$；输入含 **物体状态与参考轨迹**；优化 $r^m+r^o$（及接触项 $r^c$）。
- **训练稳定器：** (i) **点云物体跟踪奖励**（mesh 采样 $N$ 点，避免位姿权重调参）；(ii) **接触跟踪奖励**（躯干/髋/臂等链节，排除脚）；(iii) **虚拟物体 PD 力课程**（早期强辅助、增益渐衰）。
- **参考数据：** OptiTrack 同步采集人–物轨迹；人形参考经 GMR，物体轨迹直接用。
- **评测：** 四任务 Kneel / Carry / Squat / Chair；MuJoCo sim-to-sim；G1 真机（MoCap 物体状态、随机初姿、连续执行、扰动反应）。
- **主要结果（Table I，MuJoCo 均值）：** ResMimic SR **92.5%** vs GMT 直出 **10%**、从头训 **0%**、GMT 微调 **7.5%**；平均收敛 **1300** iter vs 微调 **2400**、从头 **4500**。

## 核心摘录（面向 wiki 编译）

### 1) 两阶段残差 MDP

| 阶段 | 输入 | 输出 | 优化目标 |
|------|------|------|----------|
| GMT | $s^r_t$（本体）+ $\hat{s}^r_t$（参考，含未来窗） | $a^{\mathrm{gmt}}_t$ | $\mathbb{E}[\sum \gamma^{t-1} r^m_t]$ |
| Residual | $s^r_t, s^o_t, \hat{s}^r_t, \hat{s}^o_t$ | $\Delta a^{\mathrm{res}}_t$ | $\mathbb{E}[\sum \gamma^{t-1} (r^m_t + r^o_t)]$ + 接触 |

- 动作：目标关节角 → **PD 执行**；残差 actor 末层 **小增益 Xavier 初始化** 使初始 $\Delta a\approx 0$。
- GMT 观测：$s^r=[\theta,\omega,q,\dot q,a^{\mathrm{hist}}]_{t-10:t}$；$\hat{s}^r=[\hat p,\hat\theta,\hat q]_{t-10:t+10}$。

### 2) 物体与接触奖励

- **点云跟踪：** $r^o_t=\exp(-\lambda_o \sum_i \|\mathbf{P}[i]_t-\hat{\mathbf{P}}[i]_t\|_2)$，$\mathbf{P}$ 为物体 mesh 表面采样。
- **接触：** 参考轨迹 oracle 接触指示 $\hat{c}_t[i]$ × 力幅指数项 $r^c_t=\sum_i \hat{c}_t[i]\exp(-\lambda/f_t[i])$。
- **早停：** 物体点云偏离阈值；必需 body–object 接触丢失 **>10 帧**。

### 3) 虚拟物体力课程

- PD 力矩驱动物体趋近参考：$\mathcal{F}_t=k_p(\hat p^o-p^o)-k_d v^o$ 等；$(k_p,k_d)$ **渐衰**，让策略逐步接管重物体/噪声参考。

### 4) 四任务定义

| 任务 | 技能要点 |
|------|----------|
| **Kneel** | 单膝跪地抬箱；大幅下肢协调 |
| **Carry** | 箱子背到背上；负载分布变化下平衡 |
| **Squat** | 蹲起 + 臂/躯干托举；全身接触丰富 |
| **Chair** | 抬不规则重椅；实例泛化 |

### 5) 基线对照（论文 RQ）

| RQ | 结论摘要 |
|----|----------|
| Q1 GMT 能否直接做 loco-manip？ | **不能**（均值 SR 10%），但为残差提供强初始化 |
| Q2 GMT 作 base 是否更高效？ | **是**；从头训 sim-to-sim 崩溃 |
| Q3 残差 vs 微调？ | **残差更优**；微调难显式用物体状态 |
| Q4 真机？ | G1 上精确、表达力强、可抗扰；载荷至 **4.5–5.5 kg** |

### 6) 与仓库内路线的关系

| 维度 | ResMimic | OmniRetarget + holosoma | PILOT | AssistMimic |
|------|----------|-------------------------|-------|-------------|
| 核心 | GMT **残差** + 物体条件 | 交互保留 **重定向数据** | 感知 **LLC** | **双人** MARL |
| 物体 | 点云奖励 + 接触 + 虚拟力 | interaction mesh 硬约束 | 上肢残差跟踪 | partner 力交换 |
| 平台 | G1 | G1/T1 | G1 | 仿真双人 avatar |
| 机构 | Amazon FAR | Amazon FAR | 上海交大 | CMU/庆应 |

## 对 wiki 的映射

- 沉淀实体页：[ResMimic（arXiv:2510.05070）](../../wiki/entities/paper-resmimic.md)
- 交叉补强：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[holosoma](../../wiki/entities/holosoma.md)、[TWIST](../../wiki/entities/paper-hrl-stack-09-twist.md)、[VideoMimic](../../wiki/entities/videomimic.md)

## 当前提炼状态

- [x] 摘要、两阶段方法、奖励与实验要点摘录
- [x] 项目页与代码仓库三角互证
- [x] wiki 实体页与任务/流水线交叉链接规划
