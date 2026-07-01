---
type: concept
tags: [rl, sim2real, training, humanoid, policy-optimization]
status: complete
updated: 2026-07-01
summary: "Privileged Training 让 teacher 使用仿真特权信息训练，再蒸馏给真实可观测 student，是 sim2real 常见套路；蒸馏本质是把 RL 探索问题转为 Teacher 标注的监督学习。"
related:
  - ./terrain-latent-representation.md
  - ./sim2real.md
  - ../methods/imitation-learning.md
  - ../methods/dagger.md
  - ../methods/reinforcement-learning.md
  - ./domain-randomization.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-rma-rapid-motor-adaptation.md
  - ../entities/extreme-parkour.md
  - ../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md
  - ../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md
  - ../entities/dreamwaq-plus.md
  - ../entities/paper-perceptive-bfm.md
  - ../entities/paper-fada-humanoid.md
  - ../formalizations/gae.md
  - ../formalizations/mdp.md
sources:
  - ../../sources/papers/privileged_training.md
  - ../../sources/personal/perceptive_locomotion_representation_essence.md
---

# Privileged Training（特权信息训练）

**特权训练**（Privileged Training / Teacher-Student Training）：训练阶段提供给策略额外的、在真实部署时无法获取的信息，再通过知识蒸馏将能力迁移给仅使用可观测信息的部署策略。

## 一句话定义

> 让老师策略用"作弊信息"学会技能，然后教会学生策略只用真实传感器也能做到。

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Privileged Info | Privileged Information | 训练时可访问、部署时不可见的额外状态 |
| Teacher–Student | Teacher–Student Distillation | 特权教师策略蒸馏可部署学生 |
| RMA | Rapid Motor Adaptation | 从历史隐式估计环境参数的实例 |
| RL | Reinforcement Learning | 特权信息常在仿真训练阶段使用 |
| Sim2Real | Simulation to Real | 特权训练是缩小 gap 的常见技巧 |

## 为什么重要

人形/足式机器人的核心矛盾：

- **仿真里**：可以访问完整物理状态（地形高度、质量参数、摩擦系数、所有接触力……）
- **真实里**：只有 IMU、关节编码器、摄像头等有限传感器

直接只用可观测信息训练：样本效率低、策略难以收敛（状态不可完整观测）。

特权训练的价值：
1. **训练效率**：Teacher 策略用完整状态收敛快，Student 从 Teacher 蒸馏，比 Student 直接从头练快得多
2. **性能上界**：Teacher 给了 Student 一个可靠的行为模板，而不只是 reward 信号
3. **Sim2Real 关键**：避免策略依赖仿真才有的状态，天然对齐 sim2real 需求

---

## 核心机制

### 蒸馏的本质（直觉）

很多人把流程画成「Teacher → Student」，但没说明 **到底蒸馏了什么**：

| 无 Teacher | 有 Teacher |
|------------|------------|
| 深度图 → 随机动作 → 稀疏奖励 → 数十亿步探索 | 每场景 `obs → Teacher → action` 构成 **有标签数据集** |
| 本质是 RL：不知道正确答案 | Student 做 **监督学习**（动作回归 / BC），**无 PPO、无探索** |

**一句话**：Teacher 把 **「给定观测该输出什么动作」** 这个原本无标准答案的 RL 问题，转成 **已知标准答案的监督学习问题**。蒸馏的是 **Teacher 在大量场景下的决策经验**，不是网络结构或参数本身。

感知行走里常被蒸馏的是：

```text
Teacher(深度 + 高度图 + 特权状态) → 最佳动作
Student(仅深度 + 本体)             → 学习预测同一动作
```

中间 **terrain latent** 不必长得像高度图，只要足够产生正确动作（见 [地形 Latent 表征](./terrain-latent-representation.md)）。

### Teacher 能看到什么

原则：**Teacher 观测 ⊇ Student 观测**，通常 **额外** 拥有仿真特权（高度图、未来地形、接触真值等）。Teacher **也常看 Student 要用的传感器**（如深度图），避免输入空间差异过大导致蒸馏困难。

| 角色 | 典型观测 |
|------|----------|
| Student（部署） | 深度图、IMU、关节编码器 |
| Teacher（训练） | 上述全部 + 地形高度图、接触状态、仿真全局状态 |

极端变体：Teacher 仅高度图、Student 仅深度——Student 须从深度 **隐式恢复** 地形信息（Walk These Ways 的 adaptation 思路）。

### Teacher-Student 两阶段流程

```
阶段 1：训练 Teacher 策略
  输入：完整特权状态 s_priv（包含仿真才有的信息）
  训练方法：标准 RL（PPO/SAC 等）
  目标：在仿真中最大化 reward，尽量学好

阶段 2：训练 Student 策略
  输入：仅可观测状态 s_obs（传感器数据）
  目标：模仿 Teacher 的行为（行为克隆 + DAgger 变体）
  损失：KL 散度 / 动作回归 / 特征对齐
```

### Asymmetric Actor-Critic（非对称 Actor-Critic）

单阶段变体：Actor 和 Critic 使用不同的状态空间。

- **Actor**：只使用可观测状态 $s_{obs}$（直接生成部署策略）
- **Critic**：使用完整特权状态 $s_{priv}$（更准确的价值估计，指导 Actor 更新）

$$L_{actor} = -\mathbb{E}[\log \pi_\theta(a|s_{obs}) \cdot A(s_{priv}, a)]$$

优点：单阶段即可，无需两步训练；Critic 不部署，可以用所有信息。

---

## 特权信息的类型

| 类别 | 示例 | 真实部署能否获取 |
|------|------|---------------|
| 地形信息 | 高度图、斜度、障碍位置 | 需要感知（激光/深度相机），有延迟 |
| 物理参数 | 质量、摩擦系数、关节刚度 | 基本不可知，RMA 用自适应估计 |
| 接触状态 | 精确接触力、接触点位置 | 需要力传感器（部分机器人无） |
| 对手/目标状态 | 物体精确位置/速度 | 视觉估计有噪声 |
| 全局状态 | 机器人世界坐标（GPS-like） | 通常无法精确获取 |

---

## 典型算法与实现

### RMA（Rapid Motor Adaptation，Kumar et al. 2021）

最具代表性的 Teacher-Student + Sim2Real 框架。完整提炼见 **[RMA 论文实体页](../entities/paper-rma-rapid-motor-adaptation.md)**。

**阶段 1**：训练 base policy $\pi$ + encoder $\mu$
- 输入：$x_t$, $a_{t-1}$ + 特权 $e_t$（摩擦、质量、电机强度等）→ extrinsics $z_t=\mu(e_t)$
- **PPO** 联合训练，分形地形 + 生物力学奖励

**阶段 2**：训练 adaptation module $\phi$
- 输入：过去 **50 步**（0.5 s）$(x,a)$ 历史
- 目标：回归仿真中 $z_t$；用 **on-policy** rollout 数据（非仅专家轨迹）
- 损失：$\mathrm{MSE}(\hat{z}_t, z_t)$

部署时：$\phi$ @ **10 Hz** + $\pi$ @ **100 Hz** **异步**运行；无需特权信息、**无真机 fine-tuning**（A1 实机验证）。

### Learning to Walk in Minutes（ETH Zurich）

- Teacher：完整地形信息 → PPO
- Student：只用 proprioception（本体感知）→ 行为克隆
- 关键发现：高度仅凭步态历史即可隐式估计

### AMP + Teacher-Student

- Teacher：用高质量参考动作 + 判别器训练
- Student：蒸馏 Teacher 策略到本体感知策略

### RPL（人形多向深度，Zhang et al. 2026）

- **Stage 1 Teacher**：分地形 **特权高程图** + FALCON 双智能体 PPO 专家（含载荷末端力课程）。
- **Stage 2 Student**：**DAgger** 蒸馏为 **多视角深度 Transformer** 统一下身策略；部署仅前后 ZED 深度。
- 见 [RPL 实体页](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) 与 [sources/papers/rpl_arxiv_2602_03002.md](../../sources/papers/rpl_arxiv_2602_03002.md)。

### LadderMan（人形梯子攀爬，Zhao et al. 2026）

- **Stage 1 Teacher**：**状态专家** $\pi^{\text{expert}}_{\phi,z}$ 用 **hybrid motion tracking**（非对称上/下身跟踪 + 梯子接触奖励）从 **单条参考** 学到多倾角/踏棍间距攀爬；观测含梯子相对位姿等 **特权几何**。
- **Stage 2 Student**：**DAgger + PPO + KL** 蒸馏为仅 **深度 + 本体 + 攀爬方向** 的 $\pi^{\text{visual}}$；真机深度经 **VFM** 而非学生侧特权输入。
- 见 [LadderMan 实体页](../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) 与 [sources/papers/ladderman_arxiv_2606_05873.md](../../sources/papers/ladderman_arxiv_2606_05873.md)。

### DreamWaQ / DreamWaQ++（四足，Nahrendra et al.）

- **DreamWaQ（ICRA 2023）**：盲走 + **CENet** 估计隐式地形上下文；单阶段非对称 AC，不依赖显式高度图。
- **DreamWaQ++（T-RO 2026，[实体页](../entities/dreamwaq-plus.md)）**：在 DreamWaQ 上加入 **3D 点云外感知**、分层记忆与多模态 Mixer；仍用 **特权 critic + 部分观测 actor**，但 actor 输入含学习到的 $\mathbf{z}^{pe}$，实现障碍前瞻与传感器失效时的本体回退。

---

## 和标准 Sim2Real 的关系

```
标准 Sim2Real 路线：
  Domain Randomization → 在仿真中训练鲁棒策略 → 直接部署

特权训练路线：
  Teacher（特权信息）→ 高效收敛 →
  Student 蒸馏 → 仅用可观测状态 →
  Domain Randomization + 鲁棒训练 → 部署
```

两者不互斥——特权训练常和域随机化结合：Teacher 在随机参数空间训练，Student 从 Teacher 蒸馏同时也在随机参数下训练。

---

## 常见优点

- 训练效率高：Teacher 利用完整状态快速收敛
- 部署简洁：Student 只需传感器输入，不依赖特权信息
- 自然解耦：仿真能力和真实部署能力分开优化
- 可组合：与域随机化、课程学习等方法无缝结合

## 常见局限

- **两阶段训练成本**：需要训练两个网络，调试工作量翻倍
- **Student 上界有限**：Student 只能逼近 Teacher，不能超越它
- **特权信息选择**：哪些信息作为特权信息需要领域知识判断
- **Adaptation 泛化性**：Adaptation Module 对分布外情况可能失效

---

## 参考来源

- [感知 Locomotion 表征与蒸馏本质 FAQ（维护者整理）](../../sources/personal/perceptive_locomotion_representation_essence.md)
- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — ingest 档案（Kumar RMA 2021 / Lee Science Robotics 2020 / Ji 并发训练 2022）
- [sources/papers/rma_arxiv_2107_04034.md](../../sources/papers/rma_arxiv_2107_04034.md) — RMA 一手论文摘录（RSS 2021）
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — 最经典的 Teacher-Student sim2real 实现；实体页 [paper-rma-rapid-motor-adaptation](../entities/paper-rma-rapid-motor-adaptation.md)
- Zhuang et al., *Robot Parkour Learning* (2023) — Teacher-Student + 视觉输入扩展
- Cheng et al., *Extreme Parkour with Legged Robots* (ICRA 2024) — scandots + oracle 航向 → 深度 + 自预测 yaw 双重 DAgger；见 [Extreme Parkour](../entities/extreme-parkour.md)
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (Science Robotics, 2020) — 非对称 Actor-Critic 在足式机器人上的应用
- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning* (2018) — 非对称 AC 理论基础
- **ingest 档案：** [sources/papers/bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) — BFM：以 proxy agent 作 teacher，对学生做掩码在线蒸馏，把多接口 WBC 统一进 CVAE

---

## 关联页面

- [地形 Latent 表征](./terrain-latent-representation.md) — Student 深度编码向量通常不是可读高度图
- [Sim2Real](./sim2real.md) — 特权训练是 sim2real 的核心技术之一，解决训练-部署感知差异
- [Reinforcement Learning](../methods/reinforcement-learning.md) — Teacher 阶段用标准 RL 训练
- [Imitation Learning](../methods/imitation-learning.md) — Student 阶段本质上是模仿 Teacher 的行为克隆
- [Domain Randomization](./domain-randomization.md) — 常与特权训练结合，增强策略鲁棒性
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 复杂操作任务需要特权训练处理感知遮挡
- [DreamWaQ++](../entities/dreamwaq-plus.md) — 四足多模态非对称 AC 与 CENet 谱系
- [RMA](../entities/paper-rma-rapid-motor-adaptation.md) — 特权 extrinsics + 历史适应模块的经典两阶段框架
- [Extreme Parkour](../entities/extreme-parkour.md) — 四足跑酷 scandots/航向双重蒸馏范例（ROA 继承 RMA）
- [RPL](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) — 人形分地形高程专家 → 多视角深度学生
- [LadderMan](../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) — 单参考 hybrid tracking 专家 → 深度 visuomotor 学生
- [Perceptive BFM](../entities/paper-perceptive-bfm.md) — TCRS 合成 **地形一致 adapted 参考** 作盲 teacher 监督；部署仍用 **raw 参考 + 视觉学生**
- [FADA](../entities/paper-fada-humanoid.md) — 仿真特权 oracle → DAgger 蒸馏 Planner–IDM；部署仅 LoRA 微调 IDM（arXiv:2606.28476）
- [GAE（广义优势估计）](../formalizations/gae.md) — Teacher 策略训练阶段通常使用 GAE 优势估计
- [MDP](../formalizations/mdp.md) — 特权训练本质上是 MDP 中部分可观测性的一种工程解决方案

## 一句话记忆

> 特权训练让策略在仿真里"作弊"学技能，再通过蒸馏让真实部署的策略继承这些技能——训练时用全知，部署时用感知。
