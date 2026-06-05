---
type: concept
tags: [sim2real, rl, domain-randomization, deployment]
status: complete
related:
  - ../methods/reinforcement-learning.md
  - ./whole-body-control.md
  - ./safe-real-world-rl-fine-tuning.md
  - ./motion-retargeting.md
  - ./whole-body-tracking-pipeline.md
  - ../queries/cross-embodiment-transfer-strategy.md
  - ../comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md
  - ../tasks/locomotion.md
  - ./system-identification.md
  - ../methods/actuator-network.md
  - ./privileged-training.md
  - ../entities/genesis-sim.md
  - ./data-flywheel.md
  - ../queries/sim2real-gap-reduction.md
  - ../entities/gr00t-visual-sim2real.md
  - ../entities/nvidia-so101-sim2real-lab-workflow.md
  - ../entities/nvidia-physical-ai-learning.md
  - ../entities/sage-sim2real-actuator-gap-estimator.md
  - ../entities/lift-humanoid.md
  - ./humanoid-parallel-joint-kinematics.md
  - ./processor-in-the-loop-sim2real.md
  - ../methods/crisp-real2sim.md
  - ../entities/paper-slowrl-safe-lora-locomotion-sim2real.md
  - ../entities/paper-bam-extended-friction-servo-actuators.md
  - ../entities/bam-better-actuator-models.md
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ../entities/open-duck-mini.md
  - ../entities/physx-omni.md
summary: "Sim2Real 关注如何把仿真中学到的策略稳定迁移到真实机器人，是机器人学习落地的核心鸿沟。"
updated: 2026-06-04
sources:
  - ../../sources/papers/physx_omni_arxiv_2605_21572.md
---

# Sim2Real

**Sim2Real**（仿真到现实迁移）：在仿真环境训练控制策略，然后部署到真实机器人上。

## 一句话定义

在仿真里学会，在现实中生效。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真训练策略迁移到真机部署 |
| DR | Domain Randomization | 随机化仿真物理参数以提升跨域鲁棒性 |
| RMA | Rapid Motor Adaptation | 特权信息 Teacher–Student，从历史轨迹隐式估计环境参数 |
| SysID | System Identification | 标定真机动力学/摩擦等，缩小仿真–现实差距 |
| SOP | Standard Operating Procedure | 渐进式真机验证流程（吊架→空转→落地） |
| URDF | Unified Robot Description Format | 机器人运动学与惯性描述，迁移前须与真机对齐 |

## 为什么重要

- 真实机器人训练成本高、速度慢、容易损坏
- 仿真可以并行加速、任意重置、无硬件损耗
- 但仿真和现实有 domain gap，必须解决迁移问题

## Sim2Real 工程流程总览

```mermaid
flowchart TD
  R[可选 Real2Sim<br/>CRISP 等：视频 → 可仿真资产]
  A[资产与模型准备<br/>URDF/MJCF · 初值 SysID · 生成式资产验收]
  T[仿真训练<br/>DR / Curriculum / 特权·RMA 等融入本阶段]
  G[刻画 Domain Gap<br/>动力学 · 执行器 · 传感器 · 延迟 · 视觉 · 固件路径<br/>含 SAGE 等执行器 gap 画像]
  M[闭环补齐<br/>BAM/ActuatorNet · 处理器在环 · 域适应 · 中间件对齐]
  P[仿真回归与压力测试<br/>含可选 real-to-sim 评测]
  V[渐进式真机 SOP<br/>吊架 → 空转 → 落地]
  D[部署与监控]
  O[可选安全真机收尾<br/>LoRA + Recovery 等]

  R -.-> A
  A --> T --> G --> M --> P --> V --> D --> O
```

## 核心问题：Domain Gap

仿真和现实的主要差异：

- **物理参数差异**：质量、摩擦力、延迟等参数不准
- **传感器差异**：相机噪声、IMU 漂移、触觉反馈
- **动作执行差异**：电机响应延迟、控制频率限制
- **嵌入式与通信差异**：CAN/以太网抖动、线程错过周期、驱动协议路径与仿真「瞬时读写」不一致（见 [处理器在环 Sim2Real](./processor-in-the-loop-sim2real.md)）
- **视觉差异**：纹理、光照、背景

## 主要方法

### 1. Domain Randomization
在仿真中随机化物理参数，强制策略适应多样化环境。

代表工作：Tobio et al. 2018, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"

### 2. System Identification
精确测量真实机器人参数，减少仿真-现实差距。

### 3. Domain Adaptation
用视觉/感知层面的 domain adaptation 减少感知差异。

### 4. Curriculum Learning
从简单环境逐步过渡到复杂/真实环境。

### 5. Privileged Information / Teacher-Student
训练时用额外信息（如 true state、环境参数），推理时只用可观测信息。见 [Privileged Training](./privileged-training.md)。

### 6. RMA（Rapid Motor Adaptation）

**RMA** 是目前 sim2real 领域最有影响力的 Teacher-Student 框架之一（Kumar et al. 2021）。

**两阶段流程：**

```
阶段 1：训练 Base Policy（仿真中）
  输入：机器人本体感知状态 s_t + 环境特权参数 e_t（质量/摩擦/电机参数）
  算法：PPO
  产出：Base Policy π(a | s_t, e_t)

阶段 2：训练 Adaptation Module（仿真中，模拟真机条件）
  输入：过去 k 步的关节状态历史 {s_{t-k}, ..., s_{t-1}}
  目标：从历史轨迹中隐式估计 e_t 的压缩表示 ê_t
  损失：||ê_t - φ(e_t)||²（对齐 Base Policy 所用的特权嵌入）

部署（真实机器人）：
  Base Policy + Adaptation Module（无需特权参数 e_t）
```

**为什么有效**：机器人与地面的历史交互隐含了地形/摩擦/质量等信息，Adaptation Module 从行为历史中隐式重建这些信息。

**在 Unitree A1 上实测**：复杂地形（草地/碎石/台阶）零样本迁移成功。

### 7. Sim2Real SOP (标准作业程序)

根据 [xbotics-embodied-guide](../../sources/repos/xbotics-embodied-guide.md) 的总结，为了提高 Sim2Real 的可复现性，应遵循标准化的工程步骤：
- **前置阶段**：精确的 URDF 建模与动力学参数初步对齐；若场景物体来自 **生成式 sim-ready 管线**（如 [PhysX-Omni](../entities/physx-omni.md) 导出的 URDF/XML），须单独验收 **惯性、碰撞盒与关节轴** 是否与目标仿真器一致，不宜默认「生成即可用」。
- **仿真验证**：在 [isaac-gym-isaac-lab](../entities/isaac-gym-isaac-lab.md) 或 [genesis-sim](../entities/genesis-sim.md) 中完成基础策略训练，并通过域随机化覆盖物理参数偏差。
- **评测基础设施**：产业侧亦将可信仿真用于 **real-to-sim 闭环排序**（训练仍主要来自真机），见 [仿真评测基础设施](simulation-evaluation-infrastructure.md) 与 [Genesis World 1.0](../entities/genesis-world-10.md)。
- **中间件对齐**：统一仿真与真机的控制频率（如 50Hz 策略 + 200Hz 关节 PD）与动作/状态归一化标准。
- **实物测试**：采用“吊架测试 -> 空转测试 -> 落地测试”的渐进式 SOP。

### 8. 高保真执行器对齐 (Actuator Alignment)

**解析摩擦扩展（舵机 / 伺服）：** [BAM](../entities/paper-bam-extended-friction-servo-actuators.md)（ICRA 2025，[Rhoban/bam](https://github.com/Rhoban/bam)）在 MuJoCo 等默认 Coulomb–Viscous 之外，用 **M1–M6 可辨识摩擦上界**（Stribeck、负载相关、谐波二次项）与摆锤台架 **CMA-ES** 标定，在 Dynamixel / eRob 2R 臂上可将轨迹 MAE 降至约一半——尤其适合 **RL 低 PD 增益** 下执行器滞后明显的场景；与 [Actuator Network](../methods/actuator-network.md)（数据驱动）及 [SAGE](../entities/sage-sim2real-actuator-gap-estimator.md)（gap 度量）可组合使用。

根据 [zest](../methods/zest.md) 的实践，缩小动力学差距的关键在于精确处理闭链执行器（如膝盖、脚踝）的物理特性。通过基于电枢（Armature）分析值的增益选择程序，可以在不使用反馈补偿器的情况下，实现高动态动作的零样本迁移。机构层闭链几何、驱动—关节力映射与「训练用开环树 / 真机串并联」落差，可对照 [人形机器人并联关节解算](./humanoid-parallel-joint-kinematics.md)（含 LiPS、Kinematic Actuation Models 等文献锚点）。

### 9. 处理器在环（固件 + 外设路径）

当失效主要来自 **固件调度、总线语义与传感器融合实现** 而非刚体参数本身时，可在仿真中运行**未改动的生产固件**，并用 I2C/CAN 等外设仿真注入寄存器级数据流与请求–响应抖动，使 RL 策略与底层栈在同一闭环里被联合测试。工程动机与管线拆分见 [处理器在环 Sim2Real](./processor-in-the-loop-sim2real.md)。

## 常见误区

- **以为仿真越逼真越好**：太精确的仿真不一定更好，domain randomization 可能更 robust
- **忽略动作延迟**：仿真中动作瞬时执行，现实中有延迟
- **只看 reward 不看安全性**：sim2real 部署初期容易损坏硬件

## 在人形机器人中的应用

人形机器人 sim2real 的特殊挑战：

- 高维状态空间（30+ 自由度）
- 接触力难以精确建模
- 视觉感知差异大
- 足式接触的不确定性

典型 pipeline：

```
仿真训练 → 域随机化 → 零样本迁移 → 真实机器人部署 → 在线微调（可选）
```

### 在「映射 → 训练 → 迁移」三段流水线中的位置

人形动作落地的整条链是「**映射 → 训练 → 迁移**」三段：[重定向流水线](./motion-retargeting.md) 把人体参考映射成物理可执行参考（**映射**），[WBT 流水线](./whole-body-tracking-pipeline.md) 把这些参考当训练数据学出全身跟踪策略（**训练**），[跨具身策略迁移](../queries/cross-embodiment-transfer-strategy.md) 再把策略搬到新机体（**迁移**）。Sim2Real **横切训练与迁移两段**：它既决定阶段 ② 训练出的策略能否零样本上真机，也决定阶段 ③ 换机体后是否需要重新跨越 domain gap。本页的 RMA / 域随机化 / 执行器对齐 / 安全 LoRA 收尾正是这条链「从仿真策略到真机可执行」的关键工程手段——其中 [SLowRL](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md) 式「冻结策略 + rank-1 LoRA + 安全壳」尤其契合**跨具身迁移后**的真机收尾。不同 WBT 方法（[SONIC / BeyondMimic / SD-AMP / Heracles 对比](../comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md)）在「仿真训练 vs 真机微调」的比重上各有取舍。

> 部署后的真机在线适配自成一段窄口议题（低秩残差 / 生成兜底 / CBF 安全壳三条路径），单列于 [真机安全 RL 微调](./safe-real-world-rl-fine-tuning.md)。

- **安全、参数高效的真机微调（四足）：** [SLowRL](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md)（arXiv:2603.17092）在 **冻结仿真策略** 上只训 **rank-1 LoRA**，并用 **Recovery Policy + Safety Filter** 约束真机探索；Unitree Go2 jump/trot 上相对全参 PPO 微调约 **46.5%** 墙钟缩短、训练期摔倒近零，适合讨论「**不全参、不盲探索**」的 sim2real 收尾阶段。

- **补充参照（学习式管线）：** [LIFT](../entities/lift-humanoid.md) 将「预训练期高随机性探索」与「微调期真机侧确定性动作」拆开，并把随机探索主要约束在 **物理知情世界模型** 的 rollout 中，用于讨论 **安全–样本效率** 折中；其站点亦给出 **预训练任务设计不当 → 零样本 sim2real 失败**、再靠短时段实机数据恢复的案例叙事。

- **补充参照（低成本双足 / 舵机）：** [Open Duck Mini](../entities/open-duck-mini.md) 在 **Feetech 舵机 + BAM 电机辨识 + MuJoCo Playground** 管线上公开 sim2real 行走；强调 MJCF 执行器参数与真机一致、模仿奖励与参考运动分仓迭代，机载部署在 Pi Zero 2W（见 [Open Duck Mini Runtime](../entities/open-duck-mini-runtime.md)）。

- **补充参照（人形 loco-manip · 冻结策略适配）：** [SplitAdapter](../entities/paper-splitadapter-load-aware-loco-manipulation.md)（arXiv:2606.03297）在 **冻结 AMP 搬箱策略** 上学习 **物体/负载** 与 **动力学** 双分支历史适配（分裂世界模型 + GRL + 分层 FiLM），针对 **载荷与搬放高度变化** 与 **sim–real 动力学差** 的耦合；MuJoCo sim-to-sim 与 **Unitree G1 零样本** 重载（6 kg）全流程成功率显著提升，可与 RMA 式「单 latent 外参估计」对照阅读。

### Real2Sim：从视频构造可仿真资产

讨论 Sim2Real 时常隐含「仿真里已有合理关卡与参考运动」；人形上下文技能还要解决如何把**单目视频**变成**接触动力学可信**的仿真资产。[CRISP](../methods/crisp-real2sim.md)（ICLR 2026）用**凸平面场景原语 + 人–场景接触补全 + RL 人形闭环**把视频推向可 rollout 的 Real2Sim，并与 VideoMimic 等管线在几何—控制接口上形成对照（见项目页交互对比区）。

## 参考来源
- [KungFuAthleteBot](../../sources/papers/kung_fu_athlete_bot.md)

- Tobin et al. 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* — domain randomization 奠基论文
- Peng et al. 2018, *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* — locomotion 控制迁移基线
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — DR / RMA / InEKF ingest 摘要
- [Sim2Real 论文导航](../../references/papers/sim2real.md) — 论文集合
- [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/) — 工程实践
- [机器人论文阅读笔记：Domain Randomization](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/Domain_Randomization_Understanding_Sim-to-Real_Transfer/Domain_Randomization_Understanding_Sim-to-Real_Transfer.html)
- [机器人论文阅读笔记：LCP](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/LCP_Sim-to-Real_Action_Smoothing/LCP_Sim-to-Real_Action_Smoothing.html)
- [机器人论文阅读笔记：RMA](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_Sim-to-Real/RMA_Rapid_Motor_Adaptation/RMA_Rapid_Motor_Adaptation.html)
- [Menlo：Noise is all you need…](../../sources/blogs/menlo_noise_is_all_you_need.md) — 处理器在环 + CAN 抖动注入的 Asimov 工程博文入库摘录
- **ingest 档案：** [sources/courses/nvidia_sim_to_real_so101_isaac.md](../../sources/courses/nvidia_sim_to_real_so101_isaac.md) — NVIDIA SO-101 动手课：DR / Co-training / Cosmos / SAGE+GapONet 四类策略对照与 VLA workflow
- **ingest 档案：** [sources/repos/sage-sim2real-actuator-gap.md](../../sources/repos/sage-sim2real-actuator-gap.md) — SAGE：Isaac Sim 重放与真机日志对齐的执行器层 sim2real gap 度量工具链
- [sources/papers/crisp_real2sim_iclr2026.md](../../sources/papers/crisp_real2sim_iclr2026.md) — CRISP：单目视频平面原语 Real2Sim + 接触引导（ICLR 2026）ingest 摘录
- **ingest 档案：** [sources/papers/barkour_arxiv_2305_14654.md](../../sources/papers/barkour_arxiv_2305_14654.md) — Barkour：>1m/s 敏捷动作的额外 DR + 零样本 sim2real 完成 5m×5m 障碍课
- **ingest 档案：** [sources/papers/slowrl_arxiv_2603_17092.md](../../sources/papers/slowrl_arxiv_2603_17092.md) — SLowRL：LoRA + Recovery 安全真机微调（Go2）
- **ingest 档案：** [sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md](../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md) — BAM：舵机扩展摩擦模型 + MuJoCo 2R 验证（arXiv:2410.08650，ICRA 2025）
- **ingest 档案：** [sources/repos/rhoban_bam.md](../../sources/repos/rhoban_bam.md) — Rhoban/bam 开源辨识与仿真管线

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [真机安全 RL 微调](./safe-real-world-rl-fine-tuning.md) — 部署后真机在线适配的安全边界：低秩残差 / 生成兜底 / CBF 安全壳三条路径
- [Motion Retargeting](./motion-retargeting.md) — 「映射 → 训练 → 迁移」三段流水线首段：Sim2Real 消费其物理可执行参考产物
- [Whole-Body Tracking Pipeline](./whole-body-tracking-pipeline.md) — 三段流水线中段；Sim2Real 横切其「训练 → 真机」落地
- [跨具身策略迁移选型指南](../queries/cross-embodiment-transfer-strategy.md) — 三段流水线末段；换机体后是否需重跨 domain gap
- [Locomotion](../tasks/locomotion.md)
- [System Identification](./system-identification.md)（减少物理参数和执行器模型的 sim2real gap）
- [Actuator Network 执行器网络](../methods/actuator-network.md) — 用神经网络拟合电机非线性特性
- [Privileged Training](./privileged-training.md)（Teacher-Student 训练是 sim2real 的核心技术之一）
- [Query：RL 策略真机调试 Playbook](../queries/robot-policy-debug-playbook.md) — 真机部署阶段系统排障指南
- [LEGS（论文实体）](../entities/paper-legs-embodied-gaussian-splatting-vla.md) — 3DGS 缩小 **视觉** sim2real gap 以合成 VLA 训练数据（arXiv:2606.01458）
- [NVIDIA SO-101 Sim2Real 实验 workflow](../entities/nvidia-so101-sim2real-lab-workflow.md) — 官方动手课：四类 sim2real 策略 + GR00T N1.6 VLA + LeRobot/Isaac Lab
- [GR00T-VisualSim2Real](../entities/gr00t-visual-sim2real.md) — NVIDIA 视觉 Sim2Real 框架，PPO Teacher + DAgger RGB Student，Unitree G1 零样本迁移（CVPR 2026）
- [SAGE（执行器 Sim2Real 间隙估计）](../entities/sage-sim2real-actuator-gap-estimator.md) — Isaac 重放与真机关节日志对齐，RMSE/相关/余弦相似度等量化执行器层 gap
- [LIFT](../entities/lift-humanoid.md) — JAX SAC 大规模预训练 + Brax 物理知情世界模型微调；微调阶段真机确定性采集与模型内随机探索解耦（arXiv:2601.21363）
- [人形机器人并联关节解算](./humanoid-parallel-joint-kinematics.md) — 并联踝闭链与仿真训练接口分层（冲击下传载再分配等）
- [处理器在环 Sim2Real](./processor-in-the-loop-sim2real.md) — 固件/总线/调度纳入仿真闭环的腿式迁移路径
- [CRISP（Contact-guided Real2Sim）](../methods/crisp-real2sim.md) — 单目视频 → 凸平面场景原语 + 接触补全 → RL 物理闭环的 Real2Sim（ICLR 2026）
- [SLowRL（安全 LoRA 真机微调）](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md) — 四足动态策略的低秩 + Recovery 安全层
- [BAM 扩展摩擦（舵机仿真）](../entities/paper-bam-extended-friction-servo-actuators.md)、[BAM 开源仓库](../entities/bam-better-actuator-models.md) — M1–M6 摩擦辨识与 MuJoCo 2R 验证

## 继续深挖入口

如果你想沿着 sim2real 继续往下挖，建议从这里进入：

### 论文入口
- [Sim2Real 论文导航](../../references/papers/sim2real.md)

### 仿真 / 平台入口
- [Simulation](../../references/repos/simulation.md)
- [RL Frameworks](../../references/repos/rl-frameworks.md)

## 推荐继续阅读

- [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/)
- [SAGE 官方仓库 README](https://github.com/isaac-sim2real/sage)（执行器层 gap 度量与成对数据集管线）
- [Query：如何缩小 sim2real gap](../queries/sim2real-gap-reduction.md)
- [Comparison：Sim2Real 方法横向对比](../comparisons/sim2real-approaches.md)
