---
type: method
tags: [imitation-learning, privileged-training, dagger, teacher-student, sim2real]
status: complete
updated: 2026-11-07
summary: "飞书 Know-How 模块：Teacher-Student 用仿真特权信息训练教师，再蒸馏或 DAgger 聚合数据训练可部署学生，缓解 BC 分布偏移与 sim2real 观测差距。"
related:
  - ./multi-expert-distillation.md
  - ./dagger.md
  - ../concepts/rl-runner.md
  - ../concepts/privileged-training.md
  - ./behavior-cloning.md
  - ./imitation-learning.md
  - ../concepts/sim2real.md
  - ../overview/humanoid-rl-motion-control-methods.md
  - ../entities/paper-humanoid-touch-dream.md
  - ../entities/paper-cts-concurrent-teacher-student-locomotion.md
original:
  - ../../sources/papers/dagger_reading_arxiv_1011_0686.md
  - ../../sources/papers/learning_by_cheating_arxiv_1912_12294.md
sources:
  - ../../sources/papers/dagger_reading_arxiv_1011_0686.md
  - ../../sources/papers/learning_by_cheating_arxiv_1912_12294.md
  - ../../sources/papers/learning_quadrupedal_locomotion_challenging_terrain_arxiv_2010_11251.md
  - ../../sources/personal/rl_runner_types.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
  - ../../sources/papers/privileged_training.md
---

# Teacher-Student 与 DAgger 训练

RoboParty 飞书 Know-How 将 **Teacher-Student 模型** 与 **DAgger（Dataset Aggregation）** 列为同一教学模块：在仿真中用**特权观测**训练强教师，再把能力迁移给仅具可部署传感器的**学生**；DAgger 进一步让学生访问自身诱导状态并由专家补标，缓解行为克隆的分布漂移。

## 一句话定义

教师「作弊」学会难技能，学生只带真机传感器跟师；DAgger 则让学生在自己会犯错的状态上向专家补课。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TS | Teacher-Student | 特权教师向可部署学生蒸馏 |
| DAgger | Dataset Aggregation | 迭代用当前策略采集状态并由专家标注 |
| BC | Behavior Cloning | 纯监督模仿，易分布偏移 |
| RL | Reinforcement Learning | 可与 TS 混合（RL 探索 + 专家修正） |
| Sim2Real | Simulation to Real | TS/DAgger 主要缩小观测与动力学 gap |
| IL | Imitation Learning | 本模块所属范式 |
| LBC | Learning by Cheating | Chen 等 2019 的两阶段特权教师→感知学生方案 |
| TCN | Temporal Convolutional Network | Lee 等 2020 学生所用的本体感受历史编码器 |

## 为什么重要

- **人形部署常态**：仿真有完美速度、高度图、接触标志；真机没有。
- **飞书代码块定位**：与 DreamWaQ、PIE 等「单阶段非对称 AC」并列，代表**两阶段或迭代 IL** 路线。
- **与 BFM 多技能 TS** 区分：本页是**通用训练范式**；多动作 BFM 见 [teacher-student-multi-skill-bfm](./teacher-student-multi-skill-bfm.md)。
- **并发 TS 对照：** [CTS（IEEE RA-L 2024 / arXiv:2405.10830）](../entities/paper-cts-concurrent-teacher-student-locomotion.md) 将 teacher/student **同一 PPO 循环联合训练**，相对本页两阶段蒸馏在盲 locomotion 上报告更高跟踪精度；官方项目页无 GitHub，工程实现见 [Legbot Lab](../entities/legbot-lab.md)。

## 核心原理

**Teacher-Student（蒸馏）：**
1. 训练 $\pi_{\mathrm{teacher}}(a \mid s_{\mathrm{priv}})$
2. 收集 $(s_{\mathrm{deploy}}, a^*)$ 其中 $a^* \sim \pi_{\mathrm{teacher}}(\cdot \mid s_{\mathrm{priv}}(s_{\mathrm{deploy}}))$
3. 监督训练 $\pi_{\mathrm{student}}(a \mid s_{\mathrm{deploy}})$

注意：步骤 2 若只在教师自己的轨迹上收集 $(s_{\mathrm{deploy}}, a^*)$，本质仍是行为克隆，会遇到下面的分布偏移；用**学生自己的 rollout** 收集状态、再由教师标注，才是 DAgger 式的蒸馏。

**DAgger：**
- 每轮用 $\pi_i=\beta_i\pi^*+(1-\beta_i)\hat\pi_i$ rollout → 专家标注访问状态 → 聚合进数据集 $D$ → 在整个 $D$ 上重训 $\hat\pi_{i+1}$，最后返回验证集上最好的策略（详见 [DAgger](./dagger.md)）。常用 $\beta_1=1$（无需初始策略），最简版本 $\beta_i=\mathbb{1}(i=1)$。

### 为什么需要 DAgger：分布偏移的量级（Ross 等 2011）

行为克隆在专家状态分布上训练，若单步出错概率为 $\epsilon$，在自身诱导分布下 $T$ 步内的额外代价可达 $T^2\epsilon$（论文称该界紧）；DAgger 在假设损失强凸有界等条件下，把额外代价降到随 $T$ 线性（$uT\epsilon_N+O(1)$，$u$ 为单步偏离的最大代价增量）。直觉：一旦出错就会进入示教里从没见过的状态，而 DAgger 让专家在这些状态上补标。

### 机器人/驾驶里的特权教师（Chen 等 2019 *Learning by Cheating*）

- 先训"作弊"的特权智能体（读真值鸟瞰图与交通参与者位置），再让它当教师训练只用单目相机的学生；
- 教师是"白盒"：可在**任意**状态被查询，并一次给出所有高层指令（左转 / 右转 / 直行 / 跟车道）下的动作，因此天然支持 DAgger 式的在线监督与并行监督学生所有分支；
- 训练流程：专家轨迹训教师 → 离线行为克隆学生 → DAgger 在线训练学生（只做在线阶段最终精度一样，但更慢，所以预训加速）；
- 结果：CARLA 原基准所有任务 100% 成功，NoCrash 最难条件比此前最好高 18 个百分点；消融表明"白盒监督"与"在线轨迹"各自都显著提升性能。

腿足上的对应做法见 Lee 等 2020（Science Robotics，本次仅部分核对）：直接用 RL 训练粗糙地形盲走"没有成功"，改为先用 RL（TRPO）训练读取地形与接触真值的特权教师，再用 DAgger 训练只用本体感觉历史（TCN）的学生；学生损失是动作与潜向量两项平方误差。

## 主要技术路线

| 路线 | 代表链接 | 说明 |
|------|----------|------|
| 迭代 IL | [DAgger](./dagger.md) | 分布聚合纠偏 |
| 特权信息 | [Privileged Training](../concepts/privileged-training.md) | 仿真作弊观测 |
| Sim2Real | [Sim2Real](../concepts/sim2real.md) | 部署 gap 主线 |

## 工程实践

- 明确 **特权列表**（地形高度、真实摩擦、外力）与 **部署列表**（IMU、关节、可选深度）。
- 记录每轮 DAgger 的 **状态覆盖** 与失败模式，避免只在专家分布附近重复采样。
- 与单阶段非对称 AC 对比样本效率与调试成本（飞书强调两种路线并存）。
- **教师必须能在任意状态被查询：** 这是 DAgger 式蒸馏的前提；仿真里的教师策略满足，真实专家则不行。
- **多头/多指令教师：** Learning by Cheating 的做法是让教师一次输出所有指令分支，一个状态上同时监督学生所有分支。
- **混合系数：** 若教师可靠，可直接令 $\beta_i=\mathbb{1}(i=1)$（只在首轮用教师动作），其余轮用学生自己的动作采样。

## 局限与风险

- **专家质量上限**：教师本身不强，学生无法超越。
- **仿真特权泄露**：若特权与部署观测存在确定性映射漏洞，可能导致过拟合作弊特征。
- **工程迭代成本**：DAgger 需专家在线或自动标注管线。
- **DAgger 的理论前提：** Ross 等的界假设损失强凸有界并需要无悔的在线学习器；神经网络学生并不满足这些假设，保证只能当作直觉而非定量预测。
- **教师信息过强：** 学生可能无法从自己的传感器恢复教师用到的特权量（部分可观测，见 [POMDP](../formalizations/pomdp.md)），此时蒸馏误差不会消失。
- **本页证据范围：** 数值与结论来自 DAgger 论文和 Learning by Cheating（驾驶，非机器人本体）；腿足部分的 Lee 等工作只部分核对，其余机器人案例（HTD、CTS 等）转引自其他页面。

## 关联页面

- [Multi-Expert Distillation（多专家蒸馏）](./multi-expert-distillation.md) — 多 RL 专家 → DAgger → 单策略（locomotion 主线）
- [DAgger](./dagger.md)、[Privileged Training](../concepts/privileged-training.md)
- [RL Runner（训练循环编排）](../concepts/rl-runner.md) — Distillation Runner vs Imitation Runner：Teacher 网络 vs 专家演示
- [Teacher-Student 多技能 BFM](./teacher-student-multi-skill-bfm.md)
- [POMDP](../formalizations/pomdp.md) — 学生看不到特权量时的部分可观测视角
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)
- [HTD 解耦 WBC](../entities/paper-humanoid-touch-dream.md) — 开源 BC 250k → DAgger 至 600k 的 G1 下肢蒸馏日程
- [CTS 并发 Teacher–Student](../entities/paper-cts-concurrent-teacher-student-locomotion.md) — 单阶段 PPO 联合优化（Legbot Lab 工程扩展）
- [Fault-Tolerant Locomotion](../entities/paper-fault-tolerant-locomotion.md) — 单阶段非对称 actor–critic + latent-alignment（对照两阶段蒸馏）

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)
- [privileged_training.md](../../sources/papers/privileged_training.md)
- [DAgger 论文归档（原文通读）](../../sources/papers/dagger_reading_arxiv_1011_0686.md)
- [Learning by Cheating 论文归档（原文通读）](../../sources/papers/learning_by_cheating_arxiv_1912_12294.md)
- [Lee 等 2020 论文归档（部分核对）](../../sources/papers/learning_quadrupedal_locomotion_challenging_terrain_arxiv_2010_11251.md)

## 推荐继续阅读

- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (2011) — <https://arxiv.org/abs/1011.0686>
- Chen et al., *Learning by Cheating* (CoRL 2019) — <https://arxiv.org/abs/1912.12294>
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (2020) — <https://arxiv.org/abs/2010.11251>
- [Sim2Real](../concepts/sim2real.md)
