# LHBS: Learning Human-Like Badminton Skills for Humanoid Robots（arXiv:2602.08370）

> 来源归档（ingest）

- **标题：** Learning Human-Like Badminton Skills for Humanoid Robots（**LHBS**）
- **类型：** paper / humanoid / badminton / AMP / DAgger / goal-conditioned RL / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2602.08370>
- **arXiv HTML：** <https://arxiv.org/html/2602.08370v1>
- **PDF：** <https://arxiv.org/pdf/2602.08370>
- **项目页：** <https://astrorix.github.io/LHBS/>
- **机构：** 香港大学（HKU）、EngineAI（深圳）
- **硬件：** EngineAI PM01 人形；真机评测辅以 FZMotion 光学动捕获取基座 6-DoF 与羽毛球 3D 位置
- **仿真栈：** NVIDIA Isaac Sim + Isaac Lab；物理 200 Hz、策略 50 Hz
- **入库日期：** 2026-06-25
- **一句话说明：** **Imitation-to-Interaction** 四阶段渐进 RL——MoCap 教师模仿 → DAgger 目标条件蒸馏 → AMP 风格稳定 → 羽毛球物理交互 + **流形扩展**；PM01 真机 **零样本** 完成正/反手挑球等拟人击球。

## 摘要级要点

- **问题：** 羽毛球 = 爆发全身协调 + **时序关键** 拦截；纯运动模仿难以同时保证 **击球功能性** 与 **动作自然度**。
- **框架：** 从「mimic」演进到「striker」——先建运动先验，再压缩观测、用对抗先验稳态，最后在物理交互环境中把稀疏击球点 **扩展为稠密时空流形**。
- **四阶段：** (1) 教师跟踪 MoCap；(2) DAgger 蒸馏到 **本体 + 任务目标 + time-to-hit** 的紧凑状态；(3) RL + AMP 风格奖励抑制漂移；(4) 仿真羽毛球动力学 + 流形扩展 + 节奏随机化。
- **技能：** 反手/正手挑球（lift）、吊球（drop shot）等；仿真 easy/hard 双难度。
- **真机：** 据作者称 **首个** 拟人羽毛球技能 **零样本 sim2real**；受控 10 次试验：正手挑球 SR **90%**、反手挑球 **70%**。

## 核心摘录（面向 wiki 编译）

### 四阶段管线（项目页 / 论文 Fig. 2）

| 阶段 | 名称 | 观测 / 监督 | 要点 |
|------|------|-------------|------|
| 1 | Imitation | 本体（蓝）+ 模仿目标（绿） | 教师策略鲁棒跟踪 **优化重定向** 后的人形 MoCap |
| 2 | Distillation | 本体 + 任务目标（黄：击球/恢复）+ **time-to-hit**（红） | **DAgger** 去掉对未来轨迹的依赖 |
| 3 | Stabilization | 同上 + AMP 判别器 | 风格奖励 + 跟踪误差，抑制动力学漂移 |
| 4 | Interaction | 物理羽毛球 + 扩展流形 | 稀疏演示 → 稠密 **interaction volume**；含球物理与节奏随机化 |

### 仿真定量（Lift 任务，Table II）

| 方法 | SR↑ easy/hard | MSE↓ easy/hard | IBR↑ easy/hard |
|------|---------------|----------------|----------------|
| **Ours** | **0.9516 / 0.9153** | **0.0062 / 0.0108** | 0.1999 / **0.1575** |
| w/o Stab. | 0.9501 / 0.9147 | 0.0094 / 0.0162 | 0.1363 / 0.1148 |
| E2E-AMP | 0.8586 / 0.7391 | 0.2109 / 0.6004 | **0.2383** / 0.0990 |
| w/o Interact. | 0.3787 / 0.3583 | 0.3419 / 0.3569 | -0.0210 / -0.0230 |
| ASE-Based | 0.0037 / 0.0034 | 4.27 / 4.34 | -0.27 / -0.27 |
| VQ-Based | 0.0042 / 0.0041 | 3.31 / 3.38 | -0.28 / -0.28 |

- **SR**：成功拦截并触球比例；**MSE**：空间跟踪误差；**IBR**：回球质量（含过网等约束）。
- **消融：** 去掉 Stage 3 风格稳定 → MSE 变差；去掉 Stage 4 交互 → SR 崩溃，说明稀疏 MoCap **不足以** 覆盖动态拦截流形。
- **E2E-AMP** IBR 偶高但 SR/MSE 差——作者归因 **幸存者偏差**（只打到易回界的球）。

### 与相邻人形球类技能对照

| 维度 | LHBS（本文） | [LATENT](../../wiki/entities/paper-notebook-latent.md) 网球 | [Whole-Body Badminton](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md) |
|------|-------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 平台 | EngineAI **PM01** | Unitree **G1** | （待深读） |
| 核心难点 | 模仿 → **物理击球交互** | 不完美 MoCap → **latent 修正** | 多阶段 RL（待补充） |
| 先验 | AMP + DAgger 蒸馏 | Latent action + **LAB** 屏障 | — |
| 泛化 | **流形扩展** 稠密击球点 | 连续对打组合 | — |
| Sim2Real | **零样本** 挑球 | 真人对打多拍 | — |

## 对 wiki 的映射

- 升格实体页：[LHBS（Paper Notebooks #04）](../../wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md)
- 交叉：[amp-reward](../../wiki/methods/amp-reward.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[LATENT](../../wiki/entities/paper-notebook-latent.md)、[sim2real](../../wiki/concepts/sim2real.md)

## 参考来源（原始）

- arXiv:2602.08370 — 论文正文
- 项目页：<https://astrorix.github.io/LHBS/>
- [humanoid_pnb_learning-human-like-badminton-skills-for-humanoi.md](humanoid_pnb_learning-human-like-badminton-skills-for-humanoi.md)
