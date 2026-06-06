---
type: formalization
tags: [safe-rl, lora, fine-tuning, sim2real, optimization, projection, math]
status: complete
created: 2026-06-03
updated: 2026-06-03
related:
  - ../concepts/safe-real-world-rl-fine-tuning.md
  - ../entities/paper-slowrl-safe-lora-locomotion-sim2real.md
  - ./cmdp.md
  - ../concepts/safety-filter.md
  - ../concepts/control-barrier-function.md
  - ../concepts/sim2real.md
  - ../methods/safe-rl.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/slowrl_arxiv_2603_17092.md
summary: "安全 LoRA 投影更新形式化：把真机微调写成『冻结仿真权重 + 低秩残差』的约束策略优化，安全约束以参数侧 LoRA 子空间限制 + 动作侧投影算子 ΠS 双重出现；对照 SLowRL 的 rank-1 + Recovery 硬切换实现，并与全参 CMDP / 连续 QP 安全壳退化形态对照。"
---

# Safe LoRA Update Projection（安全 LoRA 投影更新形式化）

**Safe LoRA Update Projection** 是 [真机安全 RL 微调](../concepts/safe-real-world-rl-fine-tuning.md) 「低秩残差 + 显式安全层」路线（[SLowRL](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md)）的数学骨架：把「在冻结的仿真策略上抠最后几成性能、同时不在训练期摔坏硬件」这件事，写成一个**在低秩参数子空间里求解、并被两层安全投影约束**的策略优化问题。本页给出统一形式，并展示它如何退化为全参 [CMDP](./cmdp.md) 与 [连续 QP 安全壳](../concepts/control-barrier-function.md) 两种相邻形态。

## 一句话定义

冻结仿真权重 $W_0$，只在低秩残差 $\frac{\alpha}{r}BA$ 上做约束策略优化，并让每一步真机动作先过一道安全投影 $\Pi_{\mathcal{S}}$ 才进执行器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| LoRA | Low-Rank Adaptation | 低秩增量微调，低成本适配大模型 |
| CMDP | Constrained Markov Decision Process | 带代价约束的 MDP，安全强化学习的标准形式 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| MDP | Markov Decision Process | 状态–动作–奖励–转移的标准序贯决策建模框架 |
| CBF | Control Barrier Function | 用前向不变集保证安全约束的控制屏障函数 |

## 决策变量与符号

| 符号 | 含义 |
|------|------|
| $W_0 \in \mathbb{R}^{d_{out}\times d_{in}}$ | 仿真预训练得到的层权重，**冻结**不更新 |
| $A \in \mathbb{R}^{r\times d_{in}},\ B \in \mathbb{R}^{d_{out}\times r}$ | LoRA 低秩因子，唯一可训练参数；秩 $r \ll \min(d_{in},d_{out})$ |
| $\alpha,\ r$ | LoRA 缩放系数与秩；有效残差为 $\frac{\alpha}{r}BA$ |
| $\Delta\theta = \{(A_\ell,B_\ell)\}_\ell$ | 所有层 LoRA 因子的集合，记策略 $\pi_{\Delta\theta}$（actor）、价值 $V_{\Delta\phi}$（critic）各自的低秩残差 |
| $\pi_{W_0}$ | 冻结仿真策略（$B=0$ 时 $\pi_{\Delta\theta}=\pi_{W_0}$） |
| $\mathcal{S}\subseteq S$ | 安全集（姿态/状态约束允许的子集） |
| $\Pi_{\mathcal{S}}$ | 安全投影算子：把主策略输出修正到不离开 $\mathcal{S}$ 的动作 |
| $\pi_r$ | 任务无关 Recovery 策略，把系统拉回名义直立低速态 $s_{nom}\in\mathrm{int}(\mathcal{S})$ |
| $c(s,a),\ \hat{c}$ | 安全代价函数与阈值（[CMDP](./cmdp.md) 口径） |

## 低秩参数化

每一层把全参微调的 $W = W_0 + \Delta W$ 限制到**秩 ≤ $r$ 的残差子空间**：

$$
h = W_0 x + \frac{\alpha}{r}\,BA\,x,\qquad B \xleftarrow{\text{init}} \mathbf{0}.
$$

$B=\mathbf 0$ 初始化保证**起步行为严格等于仿真策略**（$\Delta W=0$），微调从已知安全的工作点出发而非随机扰动。可训练参数量从 $d_{out}d_{in}$ 降到 $r(d_{out}+d_{in})$——当 $r=1$ 时约减 **99%**，把 PEFT 的低秩假设引入 50–200 Hz 级腿足闭环。

> **关键：actor 与 critic 都要低秩适配。** 若只给 actor 加 LoRA、冻结 critic，价值函数仍活在源仿真分布里，在 IsaacLab→MuJoCo/真机的分布偏移下给出错误 advantage，导致策略梯度**不收敛**（SLowRL 消融结论）。

## 约束策略优化目标

把真机微调写成在低秩子空间里求解的 [CMDP](./cmdp.md)：

$$
\max_{\Delta\theta,\ \Delta\phi}\ \ \mathbb{E}_{\tau\sim\pi_{\Delta\theta}}\!\left[\sum_{t}\gamma^t R(s_t,a_t)\right]
\quad \text{s.t.}\quad
\begin{cases}
\operatorname{rank}(\Delta W_\ell)\le r & \forall \ell \quad(\text{参数侧硬约束}) \\[4pt]
a_t = \Pi_{\mathcal{S}}\big(s_t,\ \tilde a_t\big),\ \tilde a_t\sim\pi_{\Delta\theta}(\cdot\mid s_t) & \forall t \quad(\text{动作侧投影}) \\[4pt]
\mathbb{E}\!\left[\sum_t \gamma^t c(s_t,a_t)\right]\le \hat{c} & (\text{累积代价阈值})
\end{cases}
$$

安全在这里以**两个层面**同时出现，互为补充：

1. **参数侧（隐式正则）：** 秩约束 $\operatorname{rank}(\Delta W)\le r$ 把可达策略限制在仿真行为流形附近的低维邻域，避免微调跑偏到分布外的危险行为；它不给硬安全保证，但显著缩小探索的「危险半径」。
2. **动作侧（显式护栏）：** 投影算子 $\Pi_{\mathcal{S}}$ 对每一步动作做实时安全修正，是训练期不摔倒的直接来源。

## 安全投影算子 $\Pi_{\mathcal{S}}$

$\Pi_{\mathcal{S}}$ 是「让动作不带系统离开安全集」的统一抽象，落地有两种谱系：

### 1）硬切换投影（SLowRL Recovery + Safety Filter）

$$
\Pi_{\mathcal{S}}(s,\tilde a)=
\begin{cases}
\tilde a, & s\in\mathcal{S}\ \ (\text{姿态在 pitch/roll 阈值内}) \\[4pt]
\pi_r(s), & s\notin\mathcal{S}\ \ (\text{超限}\Rightarrow\text{Recovery 覆盖})
\end{cases}
$$

这是一个**非光滑、二值**的投影：安全区透传主策略，危险区整段覆盖为 Recovery 动作把系统拉回 $s_{nom}$。$\pi_r$ 用强域随机化训练、只依赖本体感知、与具体任务（jump/trot）解耦，因此一套 recovery 可服务多个下游微调任务。

### 2）连续 QP 投影（屏障函数安全壳）

$$
\Pi_{\mathcal{S}}(s,\tilde a)=\arg\min_{a}\ \tfrac12\|a-\tilde a\|^2
\quad\text{s.t.}\quad \dot h(s,a)\ge-\gamma_h\,h(s),
$$

其中 $h$ 是 [控制屏障函数](../concepts/control-barrier-function.md)，$\mathcal{S}=\{s:h(s)\ge0\}$ 为不变集。这是 $\Pi_{\mathcal{S}}$ 的**光滑、最小修改**形态：只在接近边界处显著介入，正常区域近似恒等。

> 两种投影的取舍是「保证强度 × 模型依赖」的折中：硬切换只需本体感知阈值、零模型，但保证是经验性的；连续 QP 投影给可证明的不变性，但要可靠动力学模型与状态估计（见 [Safety Filter](../concepts/safety-filter.md)）。

## SLowRL 的具体实例化

把上面的统一形式代入 SLowRL 的设定，各符号取值如下：

| 形式化要素 | SLowRL 取法 |
|------------|-------------|
| 秩 $r$ | **rank-1**（固定真机时间预算下最优；更高秩放大接触 RL 梯度噪声） |
| 训练对象 | actor **与** critic 均加 LoRA（critic-only 冻结会不收敛） |
| 初始化 | $B=0$，起步行为 $=\pi_{W_0}$ |
| 优化器 | PPO 仅对 $\{A,B\}$ 与 actor 探索噪声求梯度 |
| $\Pi_{\mathcal{S}}$ | 硬切换：pitch/roll 超限 → Recovery $\pi_r$ 覆盖 |
| $\mathcal{S}$ 定义 | 仅本体感知（姿态角阈值），无需外部模型 |
| 报告结果 | trot/jump 上较全参 PPO 微调约 **46.5%** 墙钟缩短、训练期摔倒近零 |

低秩约束的副产物：训练期动作变化率相对全参微调在 trot 上额外下降约 **88.9%**——秩约束把策略限制在平滑的仿真行为邻域，抑制了全参探索常见的 bang-bang 高频指令。

## 退化与相邻形态对照

| 形态 | 参数侧约束 | 安全机制（$\Pi_{\mathcal{S}}$） | 何时退化到这里 |
|------|-----------|-------------------------------|----------------|
| **本页：安全 LoRA 投影**（SLowRL） | $\operatorname{rank}(\Delta W)\le r$ | 硬切换 Recovery | 仿真已能跑、真机抠收尾性能 |
| **全参 CMDP** | 无（$\Delta W$ 满秩可训） | 代价阈值 $\hat c$ 的拉格朗日罚 | 从头训练或可承受大量真机交互 |
| **纯 QP 安全壳** | $\Delta W=0$（不改策略） | 连续 QP 投影 | 已有控制栈、只加一层底线保护 |
| **生成式参考改写**（Heracles 路线） | 不改 tracker 权重 | 参考层 flow-matching 兜底 | 大扰动下要类人恢复、避免硬切换抖动 |

- 令秩约束失效（$r\to\min(d_{out},d_{in})$）且去掉 $B=0$ 暖启，本页目标退化为标准全参 CMDP。
- 令 $\Delta W=0$（完全不训练策略），只保留 $\Pi_{\mathcal{S}}$，退化为纯 [安全壳](../concepts/safety-filter.md)。
- 把硬切换 $\Pi_{\mathcal{S}}$ 换成连续 QP 投影，则得到「低秩残差 + 可证明安全壳」的叠加形态——三条路径并非互斥，实践中常叠加（见 [真机安全 RL 微调](../concepts/safe-real-world-rl-fine-tuning.md) 的三路径对比）。

## 评测口径

- **安全：** 训练期摔倒次数 / seed（SLowRL trot→0、jump→2，对照全参+安全 17.5）；触发 $\Pi_{\mathcal{S}}$ 覆盖的步数占比。
- **效率：** 回到预训练回报的墙钟时间（约 −46.5%）、真机交互步数。
- **平滑：** 动作变化率 / jerk（trot 上约 −88.9%），衡量低秩约束对高频指令的抑制。
- **秩消融：** 固定时间预算下 $r=1,2,4,\dots$ 的回报曲线，验证 rank-1 是否最优。

## 关联页面

- [真机安全 RL 微调](../concepts/safe-real-world-rl-fine-tuning.md) — 本形式化所属的「低秩残差 + 显式安全层」路线总览与三路径对比。
- [SLowRL（安全 LoRA 真机微调）](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md) — 本页的具体实现来源（rank-1 + Recovery 硬切换）。
- [Constrained MDP（CMDP）](./cmdp.md) — 约束策略优化的全参数学地基，本页是其低秩子空间退化。
- [Safety Filter](../concepts/safety-filter.md) — $\Pi_{\mathcal{S}}$ 动作侧投影的工程实现层。
- [控制屏障函数（CBF）](../concepts/control-barrier-function.md) — $\Pi_{\mathcal{S}}$ 连续 QP 投影形态的形式化基础。
- [Sim2Real](../concepts/sim2real.md) — 低秩残差所要吸收的 domain gap 来源。
- [Safe RL](../methods/safe-rl.md)、[Reinforcement Learning](../methods/reinforcement-learning.md) — 上层方法范式。

## 参考来源

- [SLowRL（arXiv:2603.17092）](../../sources/papers/slowrl_arxiv_2603_17092.md) — 冻结策略 + rank-1 LoRA（actor & critic）+ Recovery/Safety Filter 真机微调与秩消融。
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) — 低秩残差参数化范式来源。
- Altman, *Constrained Markov Decision Processes* (1999) — CMDP 约束优化框架。
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* — 连续 QP 投影 $\Pi_{\mathcal{S}}$ 的理论背景。
