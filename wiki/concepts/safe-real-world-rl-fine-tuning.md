---
type: concept
tags: [safe-rl, fine-tuning, sim2real, real-world-rl, lora, safety-filter, recovery, deployment]
status: complete
created: 2026-06-02
updated: 2026-11-14
summary: "真机安全 RL 微调：在已有 sim2real 策略上做真机在线适配时，如何用低秩残差、生成式兜底与 CBF/CLF 安全壳约束探索边界，避免训练期摔倒与硬件损坏。"
original:
  - ../../sources/papers/slowrl_arxiv_2603_17092.md
  - ../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md
related:
  - ./sim2real.md
  - ../comparisons/sim2real-vs-real2sim-fine-tuning.md
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

## 一句话定义

在真机上抠最后几成性能，但通过低秩参数子空间约束与在线安全兜底，把探索严格关进安全集里。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| CBF | Control Barrier Function | 用前向不变集保证安全约束的控制屏障函数 |
| CLF | Control Lyapunov Function | 以能量函数衰减保证稳定性的控制李雅普诺夫函数 |
| LoRA | Low-Rank Adaptation | 低秩增量微调，冻结主权重后仅优化低秩旁路矩阵 $BA$ |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| OOD | Out-of-Distribution | 分布外样本/未见状态，强扰动或硬件差异下易触发失稳 |
| QP | Quadratic Programming | 将 WBC/安全滤波写成二次规划的标准求解形式 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 为什么重要

[Sim2Real](./sim2real.md) 总览覆盖「仿真训练 → 域随机化 → 零样本迁移 → 部署」整条链；本页聚焦**部署之后的在线适配**这一窄口：

- 标准 RL 探索假设「错了可以重来」，但真机摔一次可能损坏关节、电池或外壳；全参微调（FFT）在脆弱策略上做高维探索往往需要数小时且频发机械故障。
- 高保真仿真已经消化了主要任务结构，**剩余真机残差通常是低维线性流形对齐问题**（接触求解器、执行器滞后、通信延时等），不需要推倒重训全部权重。
- 因此这一阶段的核心矛盾是 **样本效率 × 安全性**：用尽可能少的真机交互步数吸收动力学残差，同时在探索触及边界时由安全机制接管而非硬摔。

## 核心原理

把真机微调当成「在冻结的仿真行为流形上叠加受控扰动」是当前三条工程路径的共同出发点：

### 1. 低秩残差 + 显式安全层（SLowRL 路线）

[SLowRL](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md)（arXiv:2603.17092）在 IsaacLab（PhysX）完成 Stage 1 contact-explicit 预训练后，进入 Stage 2：在**冻结的预训练权重 $W_0$** 上于 **Actor 与 Critic 的全部线性层**（ELU 激活之前）并行注入低秩 LoRA 更新 $h = W_0 x + \frac{\alpha}{r}BAx$（$A\sim\mathcal{N}(0,\sigma^2),\ B=0$ 保证起点行为与预训练一致），并外挂独立安全环：

- **Actor + Critic 必须同步适配：** 消融显示仅改 Actor 时 Critic 仍锚定源仿真动力学，价值损失停在 **0.2** 且不收敛；Actor + Critic 全层适配可将价值损失降至 **0.007**（降幅 2 个数量级），同时可训练参数相对 FFT 减少 **99.09%**。
- **Rank-1 效率与平滑效应：** 在 Unitree Go2（IsaacSim → MuJoCo, 4 seeds）$\rho\in\{1,2,4,8\}$ 的 75 min / 230 min 消融中，$\rho=1$ 收敛最快且未见更高秩反超；低秩子空间抑制了 over-parameterized 微调常见的高频 bang-bang 抖动（动作变化率 $\|a_t-a_{t-1}\|$ 在 Trot/Jump 上分别下降 **88.9% / 35.7%**，FFT 仅 **38.9% / 5.7%**）。
- **Recovery Policy + Safety Filter：** 离散安全触发器每步监测本体姿态（pitch/roll 超限），超限即**硬切换覆盖**主策略输出，调用强域随机化训练的任务无关恢复策略 $\pi_r$ 将机器人拉回名义直立低速安全态 $s_{\text{nom}}$；Sim-to-Sim 摔倒均值从 FFT 无安全 **14.25 / 69.0**（Trot/Jump）及 FFT+安全 **7.5 / 17.5** 降至 **0.0 / 2.0**（收敛时间分别缩短 **38% / 55%**，均值 **46.5%**）；Unitree Go2 真机 60 min 跳跃微调平均回报从 **-19 升至 -7**（FFT 平台在 -15，接触一致性提升 **+9%**）。详见 [安全 LoRA 投影更新形式化](../formalizations/safe-lora-update-projection.md)。

### 2. 生成式兜底中间件（Heracles 路线）

[Heracles](../entities/paper-heracles-humanoid-diffusion.md)（arXiv:2603.27756）不改底层策略优化协议，而在高层参考 $\mathbf{m}_t$ 与底层通用物理 tracker 之间插入**状态条件流匹配（Conditional Flow Matching）中间件**（22.9M 参数，6 层 AdaLN-Transformer），将安全保障从「动作层硬覆盖」前移为「参考轨迹重规划」：

- **残差轨迹参数化与隐式切换：** 固定静态基线 $\bm{\beta}_{t,k}=\mathbf{p}_t$，仅预测关键帧残差 $\mathbf{r}_t$ 得 $\bm{\tau}_t=\bm{\beta}_t+\mathbf{r}_t$（$\Delta t=0.2\text{ s}$ 窗口内 $K=8$ 关键帧）。当本体状态 $\mathbf{p}_t\approx\mathbf{m}_t$ 时中间件近似恒等透传；大偏差 / OOD 下自动合成为期 5 步 Euler ODE（SDEdit directional warm start，$t_{\text{start}}=0.9$）的可行恢复关键帧并经三次样条+slerp 密化。
- **多速率分层耦合：** 在 29-DoF Unitree G1 上形成 **25 Hz** 中间件重规划（每 $N_{\text{exec}}=2$ 步）→ **50 Hz** iFSQ 离散运动表征 tracker（$f(x)=2.0\sigma(1.6x)-1$，奇数量化级含精确零状态）→ **200 Hz** 底层 PD 与物理仿真的闭环架构。
- **消融与恢复表现：** 在 MuJoCo 101 条未见序列中，Heracles 完成率 CR 达 **90.6%**（独立 iFSQ `87.2%`、VQ-VAE `86.0%`、SONIC `79.3%`）；在跌倒恢复子集中 CR 达 **90.0%**（VQ-VAE `69.8%`、iFSQ+H `52.7%`、MLP `44.0%`），且去掉训练期起始本体噪声增强后 CR 骤降至 **78.6%**（高度误差 `+91.5%`）、去掉雅可比杠杆臂加权降至 **82.1%**。

### 3. CBF / CLF 安全壳（控制论路线）

把安全约束写成可证明的**前向不变集**，用 [Safety Filter](./safety-filter.md) 在动作送入执行器前做最小二乘投影：

$$
\min_u \tfrac{1}{2}\|u - u_{\text{nom}}\|^2 \quad \text{s.t.}\ \dot{h}(x,u) \ge -\gamma h(x)
$$

其中 $h$ 是 [Control Barrier Function](./control-barrier-function.md)，可与 CLF 联合构建「稳定 + 安全」统一 QP（见 [CLF vs CBF](../comparisons/clf-vs-cbf.md)、[CLF/CBF 在 WBC 中的联合使用](../queries/clf-cbf-in-wbc.md)）。

### 三条路径对比

| 维度 | SLowRL 低秩残差 | Heracles 生成兜底 | CBF / CLF 安全壳 |
|------|-----------------|-------------------|------------------|
| 安全机制位置 | 策略外硬切换（Recovery + Filter） | 参考层生成改写 | 动作层 QP 投影 |
| 是否改策略权重 | 是（Actor+Critic 全层 LoRA） | 否（中间件外挂） | 否（滤波外挂） |
| 安全保证类型 | 经验性（本体姿态阈值） | 经验性（生成轨迹先验） | 可证明（不变集） |
| 模型依赖 | 低（本体感知 + 状态滤波） | 中（需离线训练轨迹生成器与 tracker） | 高（需动力学模型与状态估计） |
| 典型场景 | 真机在线抠性能的收尾微调 | 大扰动/跌倒下的类人恢复鲁棒性 | 已有控制栈加实时硬件底线保护 |

## 工程实践

1. **先 Sim-to-Sim 验证再上真机：** 参照 SLowRL 的 IsaacLab → MuJoCo 与 Heracles 的 IsaacLab → MuJoCo 协议，先用跨物理引擎的实时仿真验证微调超参（如 LoRA rank $\rho=1$、学习率 $10^{-2}$ vs FFT $10^{-3}$）与触发阈值，避免直接真机试错。
2. **Actor 与 Critic 全层同步加 LoRA、零初始化旁路：** 确保 $B=0$ 使微调第 0 步行为等价于 $W_0$，并在 Actor 与 Critic 每一线性层 ELU 之前并行注入 LoRA，避免价值估计错位导致策略崩溃。
3. **分离「任务微调」与「安全复位」：** 用强域随机化单独训练任务无关 $\pi_r$ 与固定安全目标态 $s_{\text{nom}}$，触发后待机器人回归稳定低速直立后再交还主策略收集下一段有效样本。
4. **注意感知依赖选型与噪声注入：** 如 Heracles 提供 38D（含全局根位置）与 35D（仅本体感知+IMU）双配置，无外置动捕时应选择本体感知闭环形态，并在生成器训练时仅对起始本体状态加入非对称高斯噪声。

## 局限与风险

1. **安全区域仍依赖人工定义（SLowRL §VII）：** $s_{\text{nom}}$ 与姿态硬阈值由人工设计；纯仿真学得的安全 critic 本身同样受 sim-to-real gap 污染，越界误判会引发误切换或漏保护。
2. **平地与中低复杂度假设（SLowRL §VII）：** SLowRL 仅在平地与中等扰动验证，楼梯/碎石等复杂非结构化地形可能需要 $\rho > 1$；且 on-policy PPO 无法复用真机历史交互轨迹。
3. **「真机微调 ≠ 从头学」与高频控制稳定性边界：** 若预训练策略 $\pi_{W_0}$ 尚未掌握基本运动结构，低秩与安全滤波只能反复复位而无法学出新技能；过激的 Safety Filter 阈值也会阻塞有效探索。
4. **外挂机制的算力与通信延迟约束：** Heracles 需在 25 Hz / 50 Hz / 200 Hz 多速率闭环下维持 5 步 ODE 推理实时性；CBF-QP 需确保高维系统每步求解严格收敛在控制周期内。

## 关联页面

- 上游与对比：[Sim2Real](./sim2real.md)、[Sim2Real 残差适配 vs Real2Sim 真机回放 vs 真机直接 RL 微调](../comparisons/sim2real-vs-real2sim-fine-tuning.md)、[Real2Sim/CRISP](../methods/crisp-real2sim.md)
- 相关实体与形式化：[SLowRL（安全 LoRA 真机微调）](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md)、[安全 LoRA 投影更新形式化](../formalizations/safe-lora-update-projection.md)、[Heracles（扩散中间件）](../entities/paper-heracles-humanoid-diffusion.md)、[LIFT](../entities/lift-humanoid.md)
- 控制与恢复基石：[Safety Filter](./safety-filter.md)、[Control Barrier Function](./control-barrier-function.md)、[CLF vs CBF](../comparisons/clf-vs-cbf.md)、[CLF/CBF 在 WBC 中的联合使用](../queries/clf-cbf-in-wbc.md)、[Balance Recovery](../tasks/balance-recovery.md)
- 同领域延伸：[CLIFT](../entities/paper-clift-closed-loop-iterative-finetuning.md)、[HIL-HARC](../entities/paper-hil-harc.md)、[Reinforcement Learning](../methods/reinforcement-learning.md)、[Query：如何缩小 sim2real gap](../queries/sim2real-gap-reduction.md)

## 参考来源

- [SLowRL（arXiv:2603.17092）](../../sources/papers/slowrl_arxiv_2603_17092.md) — 冻结全层权重 + Actor/Critic rank-1 LoRA + Recovery/Safety Filter 真机微调（`2603.17092v1` 全文核对）
- [Heracles 论文归档（arXiv:2603.27756）](../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md) — 状态条件 flow-matching 中间件 + iFSQ 物理 tracker（`2603.27756v2` 全文核对）
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* — CBF-QP 安全壳理论背景
- Hu et al., *LoRA* (2021) — 低秩微调范式来源

## 推荐继续阅读

- [SLowRL（安全 LoRA 真机微调）](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md) — 深入查看 Actor/Critic 各层 LoRA 注入与 rank 消融曲线
- [Heracles（扩散中间件）](../entities/paper-heracles-humanoid-diffusion.md) — 对比「参考层生成兜底」与「动作层硬滤波」两种安全范式
- [安全 LoRA 投影更新形式化](../formalizations/safe-lora-update-projection.md) — 低秩微调与安全集投影 $\Pi_{\mathcal{S}}$ 的统一数学表达
