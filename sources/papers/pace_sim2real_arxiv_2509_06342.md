# pace_sim2real_arxiv_2509_06342

> 来源归档（ingest）

- **标题：** Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots
- **类型：** paper
- **作者：** Filip Bjelonic, Fabian Tischhauser, Marco Hutter（ETH Zurich, Robotic Systems Lab）
- **arXiv：** <https://arxiv.org/abs/2509.06342v1>（v1，2025-09-08）
- **PDF：** <https://arxiv.org/pdf/2509.06342v1>
- **HTML：** <https://arxiv.org/html/2509.06342v1>
- **期刊投稿：** IJRR（Submitted；arXiv comment）
- **代码：** <https://github.com/leggedrobotics/pace-sim2real>
- **文档：** <https://pace.filipbjelonic.com/>
- **入库日期：** 2026-07-15
- **一句话说明：** PACE 将足式 sim2real RL 与 PMSM 物理能量模型结合：用 chirp 悬空数据 + CMA-ES 辨识紧凑关节动力学参数（$I_a,d,\tau_f,\tilde q_b,T_d$），再以四项奖励（速度跟踪、能量、碰撞、足端触地）在 Isaac Lab 盲策略训练并零样本上真机；主平台 ANYmal / Tytan / Minimal，另部署 10+ 机器人，ANYmal CoT 降至 1.27（相对 SOTA −32%），且无需动力学 DR。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2509.06342v1>
- **核心贡献：** 现有 sim2real 常依赖高维手工 reward、忽视执行器电热损耗，或需要力矩传感器 / 大规模 DR。PACE 主张 **moderate-data** 路线：用 **可物理解释的最小参数集** 对齐仿真–真机关节轨迹，再训练 **能量感知** 的紧凑 reward locomotion 策略。
- **对 wiki 的映射：**
  - [PACE 论文实体](../../wiki/entities/paper-pace-sim2real-legged-robots.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [System Identification](../../wiki/concepts/system-identification.md)

### 2) 三阶段管线与参数辨识（§2.1–2.2）

- **链接：** <https://arxiv.org/html/2509.06342v1>（Figure 3；Eq. 2–4）
- **核心贡献：**
  - **数据采集：** 固定基座、腿悬空、全关节 chirp（典型 20–60 s，0.1–2 Hz 或平台结构上限）；标准关节编码器即可，无需力矩传感。
  - **参数向量：** $\mathbf{p}=[\mathbf{I}_a,\mathbf{d},\boldsymbol{\tau}_f,\tilde{\mathbf{q}}_b,T_d]\in\mathbb{R}^{4n+1}$（每关节 armature、粘性阻尼、Coulomb 摩擦、偏置 + 全局指令延迟）。
  - **优化：** 4096 并行 Isaac 环境重放真机目标轨迹，最小化关节位置 MSE；**CMA-ES** 进化搜索（典型 ~49 维）。
- **对 wiki 的映射：**
  - [PACE 论文实体](../../wiki/entities/paper-pace-sim2real-legged-robots.md)
  - [Actuator Network](../../wiki/methods/actuator-network.md)（论文对比的黑盒基线）

### 3) 盲策略训练与四项奖励（§2.3）

- **链接：** <https://arxiv.org/html/2509.06342v1>（§2.3.2–2.3.4）
- **核心贡献：**
  - **不做动力学 DR**；仅随机化任务扰动、摩擦与地形。
  - **位置饱和** 硬件保护（近关节硬限位时 PD 目标饱和）。
  - **四项 reward：** $r_v$（基座速度跟踪）、$r_e$（PMSM 电热 + 机械 + 势能，速度归一化）、$r_{\text{ftd}}$（足端触地速度）、$r_c$（碰撞指示）；能量与 FTD 项指数调度。
  - **PPO** + 熵系数 tanh 退火；盲部署（仅本体感受）。
- **对 wiki 的映射：**
  - [Sim2Real 方法对比](../../wiki/comparisons/sim2real-approaches.md)
  - [ANYmal](../../wiki/entities/anymal.md)

### 4) 自下而上验证与跨平台部署（§3）

- **链接：** <https://arxiv.org/html/2509.06342v1>（§3.2–3.3）
- **核心贡献：**
  - **单驱动器 → 悬空整机 → 地面 locomotion** 三级评估；与 **无模型**、**Actuator Network（Hwangbo 2019）** 对比相位图（Fig.1）。
  - 主平台：**ANYmal D**、**Tytan**、**Minimal**；额外 **10+** 机器人零样本迁移。
  - **能效：** ANYmal 全 CoT **1.27**，相对先前方法 **−32%**。
- **对 wiki 的映射：**
  - [SAGE](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md)（同为执行器层 gap 工具链，栈与目标不同）
  - [BAM 扩展摩擦论文](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)（同为物理先验执行器建模）

## BibTeX（文档站提供）

```bibtex
@article{bjelonic2025towards,
  title         = {Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots},
  author        = {Bjelonic, Filip and Tischhauser, Fabian and Hutter, Marco},
  journal       = {arXiv preprint arXiv:2509.06342},
  year          = {2025},
  eprint        = {2509.06342},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## 当前提炼状态

- [x] 摘要与 §2 管线对齐 arXiv HTML
- [x] 与 pace-sim2real README / 文档站交叉索引
- [x] wiki 页面映射确认
