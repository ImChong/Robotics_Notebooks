# Evolution of Humanoid Locomotion Control（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Evolution of Humanoid Locomotion Control
- **类型：** paper / Review / Survey / humanoid locomotion control
- **期刊：** Science Robotics, 2026（Vol. 11, Issue 117；article eaed3973）
- **DOI：** <https://doi.org/10.1126/scirobotics.aed3973>
- **Science.org：** <https://www.science.org/doi/10.1126/scirobotics.aed3973>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42616832/>（PMID:42616832）
- **PDF（官方 companion）：** <https://github.com/purdue-tracelab/Humanoid-Locomotion-Survey/blob/main/Evolution_of_Humanoid_Locomotion_Control_20260819.pdf>
- **GitHub companion：** <https://github.com/purdue-tracelab/Humanoid-Locomotion-Survey>
- **阅读伴侣（作者个人页，非官方）：** <https://thejerrycheng.github.io/locomotion.html>
- **作者：** Yan Gu*、Guanya Shi*、Fan Shi*、I-Chia Chang、Yen-Jen Wang、Qilong Cheng、Zachary Olkin、Ivan Lopez-Sanchez、Yunchu Feng、Jian Zhang、Aaron D. Ames†、Hao Su†、Koushil Sreenath†（* equal；† corresponding）
- **机构：** Purdue TRACE Lab；Caltech（Ames）；UC San Diego（Su）；UC Berkeley（Sreenath）等
- **入库日期：** 2026-09-19
- **一句话说明：** Science Robotics **Review**：六十年人形 locomotion 控制从 **经典模型/优化 → 大规模仿真 RL → 生成式 foundation/world models** 三时代演进；提出 **physics-guided generative intelligence** 统一视角与三大贯穿原则（物理建模、约束决策、不确定性适应）。

## 核心论文摘录

### 1) 三时代（Fig. 1 / Table 1）

| 时代 | 考虑对象 | 工具 | 成就/目标 |
|------|----------|------|-----------|
| **Physics you can write down** | 简化/完整动力学模型 | 降阶模型、控制理论、凸优化 | ZMP 行走、Raibert hopper、HZD、DRC、Atlas parkour（经典栈） |
| **Physics you can simulate at scale** | GPU 并行仿真 + 有限真机数据 | RL、模仿学习 | 盲楼梯、跑酷、motion imitation、人形乒乓球 |
| **Physics you learn from data** | 丰富真机动力学、互联网规模数据 | Foundation models、world models、diffusion | 开放世界 loco-manipulation；**可靠性仍是开放问题** |

### 2) 统一视图（Fig. 3）

- **System 2（慢）**：目标/语义 deliberation
- **System 1（快）**：身体闭环 reactive control
- 经典层次、learned policy、foundation stack 均可落入该双层图；**变化的是哪一层被写死、哪一层被学习**。

### 3) 三大贯穿原则（结论段）

1. **Physics-based modelling** — 模型写进方程、仿真器、reward 或实时接口
2. **Constrained decision making** — 支撑多边形/力矩限 → reward/curriculum/action limit → conditioning/filter
3. **Adaptation to uncertainty** — 自适应/鲁棒控制 → domain randomization → in-context / test-time adaptation

### 4) 方法族与 companion 组织（17 families / 280 refs）

- **Modeling & classical：** LIP/ALIP/H-LIP、centroidal dynamics、convex MPC、ZMP preview、HZD、whole-body MPC、SQP/iLQR/MPPI 等
- **Learning：** Isaac Gym/Lab、MuJoCo Playground、PPO、DR/curriculum、teacher–student、privileged learning、AMASS/BeyondMimic 等
- **Emerging：** discriminative→generative、uni→multi-modal、single→multi-task、locomotion→loco-manipulation、五维 research shifts（Fig. 5）

### 5) Getting started（文内实践路线）

- **硬件：** 开源（Berkeley Humanoid/Lite、ToddlerBot）vs 商业（Unitree G1/H1、Booster、Digit、Apollo）
- **软件栈：** MIT Cheetah、OpenLoong、CasADi、acados、OCS2、Judo；Isaac Lab、MJLab、Newton/Genesis
- **RL 七步：** Isaac Sim/Lab 安装 → 仿真走策略 → 平衡/行走 → 速度条件 → human motion tracking → 感知 rough terrain → 真机部署

### 6) 开放挑战

- **安全与可靠：** 标准化 recovery/impact 测试、柔顺硬件、test-time adaptation
- **可及性：** 平台成本已降一个数量级；开源硬件/工具链决定可复现性
- **控制×认知：** 双层 hierarchy 与网络化 embodied agents 生态

## 开源核查（步骤 2.5，2026-09-19）

| 组件 | 状态 |
|------|------|
| **Companion repo** | **已开源** — PDF、Fig.1–5、分 section 阅读列表、getting-started（经典+仿真学习部分已填，其余 progressive） |
| **训练/控制代码** | **不适用** — 综述无单一可运行控制栈 |
| **阅读伴侣网页** | 第三方作者个人页（Qilong Cheng）；280 refs 可检索，**非官方 endorsement** |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-evolution-humanoid-locomotion-control.md`](../../wiki/entities/paper-evolution-humanoid-locomotion-control.md)
- 站点：[`sources/sites/purdue-tracelab-humanoid-locomotion-survey.md`](../sites/purdue-tracelab-humanoid-locomotion-survey.md)
- 仓库：[`sources/repos/humanoid-locomotion-survey.md`](../repos/humanoid-locomotion-survey.md)
- 交叉：[humanoid-rl-motion-control-body-system-stack.md](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[paper-legged-robots-advances-challenges.md](../../wiki/entities/paper-legged-robots-advances-challenges.md)（同期 SciRobotics 腿式综述对照）
