# RAVEN: Reinforcement-Adaptive Visibility-Graph Planning for Robust Humanoid Navigation with Collision-Free MPC

> 来源归档（ingest）

- **标题：** RAVEN: Reinforcement-Adaptive Visibility-Graph Planning for Robust Humanoid Navigation with Collision-Free MPC
- **类型：** paper（arXiv 预印本）
- **机构：** University of California, Los Angeles — Robotics and Mechanisms Laboratory（RoMeLa）
- **作者：** Ruochen Hou, Shiqi Wang, Beom Jun Kim, Hanzhang Fang, Mehak Singal, Dennis W. Hong
- **原始链接：**
  - <https://arxiv.org/abs/2607.15701>
  - <https://arxiv.org/html/2607.15701>
  - <https://arxiv.org/pdf/2607.15701>
- **项目页 / 代码：** 截至 **2026-07-28** **确认未开源**（arXiv abs / HTML /「Code, Data, Media」均未列官方 GitHub 或项目页；正文未承诺 code release URL）
- **发布日期：** 2026-07（arXiv）
- **入库日期：** 2026-07-21（初入库）；**2026-07-28** 复核升格（补结论、索引与导航路线交叉）
- **一句话说明：** UCLA RoMeLa 提出 **RAVEN**：用 **RL meta-policy** 在线调节 **可见图（visibility graph）障碍膨胀半径**，再由既有 **DAVG + collision-free MPC** 跟踪，并接 **Booster Gym** 底层 locomotion；在控制延迟与观测噪声下，比固定膨胀 MPC 更抗超调、比端到端 RL 更短路径且更易 sim2real；真机 **Booster T1** 半场 RoboCup 场地 mocap 验证。

## 核心论文摘录（MVP）

### 1) 问题：固定膨胀可见图 + MPC 在延迟下易超调；端到端 RL 难验安全

- **链接：** <https://arxiv.org/html/2607.15701>
- **摘录要点：** 经典 **visibility-graph + MPC** 几何最优但依赖手工膨胀；延迟、估计噪声与 locomotion 不确定性导致贴障超调。端到端 DRL 敏捷但可解释性弱，人形更需显式约束。既有「RL 调 MPC 代价权重 / 可微 MPC」计算重且对空间行为是间接杠杆；RL 出子目标 / 切换规划器则把全局几何规则当成刚性黑盒。
- **对 wiki 的映射：**
  - [RAVEN 实体页](../../wiki/entities/paper-raven-rl-adaptive-visibility-graph-mpc.md) — 问题定位。
  - [MPC vs RL](../../wiki/comparisons/mpc-vs-rl.md) — 混合选型语境。

### 2) 方法：三层栈 — RL 调图几何 → DAVG-cfMPC → locomotion

- **链接：** <https://arxiv.org/html/2607.15701>
- **摘录要点：**
  - **高层：** PPO meta-policy 输出 \(a_t\in[-1,1]^K\)，仿射映射为各障碍规划半径 \(r_{t,i}\)（膨胀），**重塑自由空间拓扑**，而非直接出速度。默认 **K=3**，actor \(d=15+5K\)，不对称 critic \(2d\)。
  - **中层：** 继承 [Hou et al., ICRA 2025] 的 **DAVG + cf-MPC**；碰撞约束在参考轨迹上线性化软约束进凸 QP（障碍半径 \(R_{obs,j}\) 即被 meta 调节的量）。
  - **低层：** 速度指令 → locomotion policy（真机用开源 **Booster Gym**，PyTorch→JIT 嵌入训练环境）。
- **对 wiki 的映射：**
  - [ARTEMIS](../../wiki/entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md) — 同实验室 **DAVG+cf-MPC** 导航骨干。
  - [Booster Gym](../../wiki/entities/paper-notebook-booster-gym-an-end-to-end-rl-framework-for-human.md) — 真机底层行走策略来源。
  - [FAR Planner](../../wiki/entities/far-planner.md) — 可见图路由对照。

### 3) 训练：Brax / MuJoCo Playground MJX + JAX cf-MPC；不对称 actor–critic

- **链接：** <https://arxiv.org/html/2607.15701>
- **摘录要点：** Actor 见含延迟/噪声观测；Critic 见特权（噪声+干净）。网络 **(512,256,128)** Swish。**1024** 并行环境、unroll **32**；RAVEN 约 **10k SPS**（MPC 是瓶颈，RTX 5090），纯 RL 约 **100k SPS**；训练步数 **1e8** vs 纯 RL **1e9** 以对齐墙钟。奖励：时间 **-5**、路径 **-20**、碰撞 **-8**、穿透 **-80**、动作率 **-0.5** + 成功 **+5000** / 摔倒 **-50000**。
- **对 wiki 的映射：**
  - [PPO](../../wiki/methods/ppo.md) — 训练算法。
  - [RAVEN 实体页](../../wiki/entities/paper-raven-rl-adaptive-visibility-graph-mpc.md) — 工程实践。

### 4) 结果：0.06 s 延迟下路径更短、穿透可控；真机轨迹更贴仿真

- **链接：** <https://arxiv.org/html/2607.15701>
- **摘录要点：**
  - **无延迟：** RAVEN 路径 **9.23 m** / 时间 **11.28 s**（MPC 9.84 / 11.43；RL 9.65 / 11.55）。
  - **0.06 s 延迟：** 固定 MPC 最大穿透 **0.128 m**、路径 **11.25 m**；纯 RL 穿透 **0** 但更慢（**12.21 s**）；RAVEN 穿透 **0.03 m**、路径 **9.33 m**、时间 **11.58 s**。
  - **真机：** T1 + mocap；RAVEN 规划侧 **~100 Hz**（cf-MPC alone ~120 Hz）；纯 RL 真机侧向振荡更大，RAVEN/MPC 仿真–真机轨迹更一致。
- **对 wiki 的映射：**
  - [Humanoid locomotion](../../wiki/tasks/humanoid-locomotion.md) — 导航层与底层行走解耦案例。
  - [MPC vs RL](../../wiki/comparisons/mpc-vs-rl.md) — 「RL 改规划几何、MPC 保约束」混合范式。
  - [导航纵深 · Stage 3](../../roadmap/depth-navigation.md) — 学习型 / 混合导航。

### 5) 开源状态（步骤 2.5）

- **结论：** 截至 **2026-07-28** **确认未开源**（无项目页、无 GitHub、正文未承诺 code release URL；arXiv「Code, Data, Media」无官方仓）。底层依赖的 **Booster Gym** 另为开源 locomotion 框架。
- **对 wiki 的映射：**
  - [RAVEN 实体页](../../wiki/entities/paper-raven-rl-adaptive-visibility-graph-mpc.md) — 局限与源码运行时序图不适用。

## 当前提炼状态

- [x] arXiv abs/HTML 摘录与数字对齐（含无延迟表、奖励权重、网络结构）
- [x] wiki 实体补 `## 结论`；与 ARTEMIS / Booster Gym / FAR / MPC–RL / humanoid locomotion / depth-navigation 交叉
- [x] `sources/README.md` 索引补录
