---
type: concept
tags: [safe-rl, fine-tuning, sim2real, real-world-rl, lora, safety-filter, recovery, deployment]
status: complete
created: 2026-06-02
updated: 2026-06-02
summary: "真机安全 RL 微调：在已有 sim2real 策略上做真机在线适配时，如何用低秩残差、生成式兜底与 CBF/CLF 安全壳约束探索边界，避免训练期摔倒与硬件损坏。"
related:
  - ./sim2real.md
  - ./safety-filter.md
  - ./control-barrier-function.md
  - ../methods/reinforcement-learning.md
  - ../entities/paper-slowrl-safe-lora-locomotion-sim2real.md
  - ../formalizations/safe-lora-update-projection.md
  - ../entities/paper-heracles-humanoid-diffusion.md
  - ../entities/lift-humanoid.md
  - ../methods/crisp-real2sim.md
  - ../comparisons/clf-vs-cbf.md
  - ../queries/clf-cbf-in-wbc.md
  - ../queries/sim2real-gap-reduction.md
  - ../tasks/balance-recovery.md
sources:
  - ../../sources/papers/slowrl_arxiv_2603_17092.md
  - ../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md
---

# Safe Real-World RL Fine-Tuning（真机安全 RL 微调）

**真机安全 RL 微调** 关心 [Sim2Real](./sim2real.md) 链路的**最后一段**：当仿真策略已经能跑、但真机上还差最后几成性能时，如何在**真实机器人上继续用 RL 调整策略**，同时**不在训练期摔坏硬件**。

## 一句话定义

在真机上抠最后几成性能，但把探索关进安全集里。

## 为什么单独立一页

[Sim2Real](./sim2real.md) 总览覆盖「仿真训练 → 域随机化 → 零样本迁移 → 部署」整条链；本页只聚焦**部署之后的在线适配**这一窄口：

- 标准 RL 探索假设「错了可以重来」，但真机摔一次可能损坏关节、电池或外壳。
- 仿真已经吃掉了大部分 domain gap，**剩余残差通常是低维的**（接触求解器、执行器滞后、实时约束），不需要全参重训。
- 因此这一阶段的核心矛盾是 **样本效率 × 安全性**：既要快速吸收真机残差，又要让训练期的危险动作有人接住。

## 残差视角：为什么不全参重训

把真机微调当成「在冻结的仿真行为流形上加一个小扰动」是这一阶段的主流工程判断：

- 仿真策略 $\pi_{W_0}$ 已经把任务结构学好，真机只需要**对齐残差**而非重新学策略。
- 全参微调（FFT）在脆弱的预训练策略上探索，容易在前期就摔倒，且会**遗忘**仿真里学到的鲁棒行为。
- 残差越低维，需要的真机交互越少、越可控——这正是下面三条路径的共同出发点。

## 三条主流路径

### 1. 低秩残差 + 显式安全层（SLowRL 路线）

[SLowRL](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md)（arXiv:2603.17092）在**冻结的仿真策略权重 $W_0$** 上只训练 **LoRA 低秩扰动** $\frac{\alpha}{r}BAx$（$B=0$ 初始化，起步行为等于仿真策略），并叠加一层显式安全机制：

- **Recovery Policy**：用强域随机化训练的任务无关策略，把机器人从可恢复状态拉回「名义直立低速」$s_{nom}$。
- **Safety Filter**：每步监测 pitch/roll 等姿态，超限则**硬切换覆盖**主策略输出。
- **关键经验**：Actor 与 Critic **都要**低秩适配（只改 Actor 不收敛，因为 Critic 仍活在源仿真分布里）；固定真机时间预算下 **rank-1 往往最优**，更高秩反而放大接触 RL 的梯度噪声。

在 Unitree Go2 的 jump/trot 上，相对全参 PPO 微调报告约 **46.5%** 墙钟缩短、训练期摔倒近零。适合「**不全参、不盲探索**」的收尾阶段。其「冻结权重 + 低秩残差 + 安全投影」可统一写成约束策略优化，见 [安全 LoRA 投影更新形式化](../formalizations/safe-lora-update-projection.md)。

### 2. 生成式兜底中间件（Heracles 路线）

[Heracles](../entities/paper-heracles-humanoid-diffusion.md)（arXiv:2603.27756）不改训练协议，而在高层参考与底层 tracker 之间插入**状态条件 flow-matching 中间件**，把「安全」做成**参考改写**而非动作覆盖：

- 名义区（本体状态 $\approx$ 参考）近似**恒等映射**，透传原参考、不损跟踪精度。
- 大偏差 / OOD 区切换为**生成器**，合成类人恢复关键帧并闭环重规划。
- **隐式模式切换**：无需手调 tracking↔recovery 的切换阈值，避免在扰动下盲目最小化即时误差产生非类人、不可恢复的扭矩。

这条路把安全保障从「策略外的滤波」前移到「参考层的生成兜底」，与 SLowRL 的「策略外硬切换」形成方法论对照。

### 3. CBF / CLF 安全壳（控制论路线）

把安全约束写成可证明的**不变集**，用 [Safety Filter](./safety-filter.md) 在动作进执行器前做最小修改投影：

$$
\min_u \tfrac{1}{2}\|u - u_{nom}\|^2 \quad \text{s.t.}\ \dot{h}(x,u) \ge -\gamma h(x)
$$

其中 $h$ 是 [Control Barrier Function](./control-barrier-function.md)，约束系统留在安全集内；可与 CLF 联合做「稳定 + 安全」的统一 QP（见 [CLF vs CBF](../comparisons/clf-vs-cbf.md)、[CLF/CBF 在 WBC 中的联合使用](../queries/clf-cbf-in-wbc.md)）。

- **优点**：理论保证强、与策略解耦、部署代价低（无需重训即可加一层安全壳）。
- **局限**：需要可靠的系统模型与状态估计；几何/规则式退化（clamp、rate limiter）实现简单但保证较弱。

## 三条路径对比

| 维度 | SLowRL 低秩残差 | Heracles 生成兜底 | CBF/CLF 安全壳 |
|------|-----------------|-------------------|----------------|
| 安全机制位置 | 策略外硬切换（Recovery + Filter） | 参考层生成改写 | 动作层 QP 投影 |
| 是否改策略权重 | 是（仅 LoRA 低秩） | 否（中间件外挂） | 否（滤波外挂） |
| 安全保证类型 | 经验性（姿态阈值） | 经验性（生成先验） | 可证明（不变集） |
| 模型依赖 | 低（本体感知阈值） | 中（需训练生成器） | 高（需动力学模型） |
| 典型场景 | 真机抠性能的收尾微调 | 大扰动下的恢复鲁棒性 | 已有控制栈加底线保护 |

三者并非互斥：实践中常**叠加**——低秩残差负责吸收 gap、生成或 Recovery 负责大偏差兜底、CBF 安全壳负责最后一道数值底线。

## 常见误区

1. **「真机微调 = 真机从头训」**：这一阶段的前提是仿真已学好任务，只需对齐**低维残差**，不是重新学策略。
2. **「LoRA 只改 Actor 就够」**：Critic 不一起低秩适配会给出源分布下的错误 advantage，导致不收敛。
3. **「有安全 critic 就够」**：纯仿真训练的安全信号本身有 sim2real 风险；显式 Recovery / 硬切换 / CBF 更可靠。
4. **「rank 越大越强」**：固定真机时间预算下 rank-1 往往最优，额外自由度主要放大接触梯度噪声。
5. **「安全壳一定让动作变保守」**：设计得好时只在接近边界处显著介入，正常区域内几乎不影响性能。

## 与相邻概念的关系

- 上游：[Sim2Real](./sim2real.md)（残差从何而来）、[Real2Sim/CRISP](../methods/crisp-real2sim.md)（用真机回放反向修仿真，减少需在真机上吸收的残差）。
- 折中参照：[LIFT](../entities/lift-humanoid.md) 把「预训练高随机探索」约束在物理知情世界模型内，微调期真机侧用确定性采集，是另一种安全–样本效率拆分。
- 下游恢复行为：[Balance Recovery](../tasks/balance-recovery.md)。

## 参考来源

- [SLowRL（arXiv:2603.17092）](../../sources/papers/slowrl_arxiv_2603_17092.md) — 冻结策略 + rank-1 LoRA + Recovery/Safety Filter 真机微调
- [Heracles 论文归档（arXiv:2603.27756）](../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md) — 状态条件 flow-matching 生成式恢复中间件
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* — CBF-QP 安全壳理论背景
- Hu et al., *LoRA* (2021) — 低秩微调范式来源

## 关联页面

- [Sim2Real](./sim2real.md) — 残差来源与整条迁移链总览
- [Safety Filter](./safety-filter.md) — 动作层最小修改的安全过滤
- [Control Barrier Function](./control-barrier-function.md) — 安全集不变性的形式化基础
- [SLowRL（安全 LoRA 真机微调）](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md)
- [安全 LoRA 投影更新形式化](../formalizations/safe-lora-update-projection.md) — 把本路线写成约束策略优化 + 安全投影 $\Pi_{\mathcal{S}}$
- [Heracles（扩散中间件）](../entities/paper-heracles-humanoid-diffusion.md)
- [LIFT](../entities/lift-humanoid.md) — 世界模型内随机探索 + 真机确定性采集
- [CRISP（Real2Sim）](../methods/crisp-real2sim.md) — 用真机回放修仿真，减少待吸收残差
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [CLF vs CBF](../comparisons/clf-vs-cbf.md)、[CLF/CBF 在 WBC 中的联合使用](../queries/clf-cbf-in-wbc.md)
- [Query：如何缩小 sim2real gap](../queries/sim2real-gap-reduction.md)
- [Balance Recovery](../tasks/balance-recovery.md)
