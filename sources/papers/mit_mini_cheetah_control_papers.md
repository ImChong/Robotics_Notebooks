# MIT Mini Cheetah / Cheetah 系控制论文（集合）

> 来源归档（ingest）

- **标题：** MIT Mini Cheetah 与 Cheetah 3 控制/动力学相关论文集合
- **类型：** paper / collection
- **入库日期：** 2026-07-25
- **一句话说明：** 支撑 [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md) 平台页的论文索引：硬件平台、Convex MPC、落地/空翻、以及 RL Rapid Locomotion。
- **关联策展：** [mit_mini_cheetah_learning_stack_curator](../personal/mit_mini_cheetah_learning_stack_curator.md)
- **执行器 thesis（已单独 ingest）：** [low_cost_modular_actuator_katz_mit_2018](./low_cost_modular_actuator_katz_mit_2018.md)

---

## 核心论文摘录

### 1) The MIT Super Mini Cheetah（Bosworth, Kim, Hogan，SSR 2015）

- **链接：** https://doi.org/10.1109/ssrr.2015.7443018
- **核心贡献：** 早期 sub-10 kg / sub-$10k 小尺度四足；足端力与阻抗轨迹实现走、跳、pronk 等。
- **对 wiki 的映射：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)

### 2) Mini Cheetah: A Platform for Pushing the Limits of Dynamic Quadruped Control（Katz, Di Carlo, Kim，ICRA 2019）

- **链接：** https://doi.org/10.1109/icra.2019.8793865
- **核心贡献：** ~9 kg / ~0.3 m 平台；模块化背驱执行器；cMPC 多步态至约 2.45 m/s；离线非线性优化 **360° 后空翻**。
- **对 wiki 的映射：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)、[paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)

### 3) MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot（Bledt et al.，IROS 2018）

- **链接：** https://doi.org/10.1109/iros.2018.8593885
- **核心贡献：** Cheetah 3 机械/腿设计与全身控制框架；与 Mini 共享大量控制思想。
- **对 wiki 的映射：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)、[gait-generation](../../wiki/concepts/gait-generation.md)、[footstep-planning](../../wiki/concepts/footstep-planning.md)

### 4) Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control（Di Carlo et al.，IROS 2018）

- **链接：** https://doi.org/10.1109/iros.2018.8594448
- **核心贡献：** SRBD + Convex MPC 力优化与足规划；后世四足 MPC 工程主流祖师爷之一。
- **对 wiki 的映射：** [srbd-convex-mpc-wbc](../../wiki/concepts/srbd-convex-mpc-wbc.md)、[mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)、[mpc.md](./mpc.md)

### 5) Rapid Locomotion via Reinforcement Learning（Margolis et al.，arXiv:2205.02824）

- **链接：** https://arxiv.org/abs/2205.02824
- **核心贡献：** Mini Cheetah 端到端 RL；报道约 **3.9 m/s**；Sim2Real、curriculum、online system identification。
- **影响叙事（策展）：** 后续 RMA、Walk These Ways、Extreme Parkour 等高速/适应工作的重要前驱之一。
- **对 wiki 的映射：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)、[sim2real](../../wiki/concepts/sim2real.md)、[paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)

### 6) Mini Cheetah, the Falling Cat（Kurtz et al.，arXiv:2109.04424）

- **链接：** https://arxiv.org/abs/2109.04424
- **核心贡献：** 空中姿态调整与落地；轨迹优化 + 机器学习案例。
- **对 wiki 的映射：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)

### 7) Real-time Optimal Landing Control of the MIT Mini Cheetah（Jeon, Kim, Kim，arXiv:2110.02799）

- **链接：** https://arxiv.org/abs/2110.02799
- **核心贡献：** 高空落地的实时接触优化 / MPC 落地控制。
- **对 wiki 的映射：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md)

## 当前提炼状态

- [x] 七条核心摘要 + wiki 映射
- [ ] 后续可选：单独升格 Rapid Locomotion / Falling Cat / Landing Control 论文实体页
