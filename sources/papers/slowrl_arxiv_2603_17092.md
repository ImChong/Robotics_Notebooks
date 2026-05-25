# SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion（arXiv:2603.17092）

> 来源归档（ingest）

- **标题：** SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion
- **类型：** paper / quadruped locomotion / sim2real / safe RL / PEFT
- **arXiv abs：** <https://arxiv.org/abs/2603.17092>
- **arXiv HTML：** <https://arxiv.org/html/2603.17092v1>
- **PDF：** <https://arxiv.org/pdf/2603.17092>
- **作者：** Elham Daneshmand, Shafeef Omar, Glen Berseth, Majid Khadiv, Hsiu-Chin Lin（机构标注含 McGill / Mila 等，见论文首页）
- **硬件：** Unitree Go2 四足（真机 jump / trot）；仿真预训练 Isaac Lab（PhysX），sim-to-sim 适配 MuJoCo
- **入库日期：** 2026-05-25
- **一句话说明：** 冻结仿真预训练策略，仅用 **LoRA 低秩适配 + 训练期 Recovery Policy 安全滤波**，在真机上安全高效微调动态运动策略；rank-1 即可恢复性能，相对全参 PPO 微调约 **46.5%** 墙钟时间缩短、近零摔倒。

## 摘要级要点

- **问题：** 运动策略 sim-to-real 后性能退化；直接在硬件上做 **全参 fine-tuning（FFT）** 样本效率低且易机械损伤。
- **方法 SLowRL：**
  - **Stage 1：** 高保真仿真预训练（contact-explicit 架构，接触目标条件化）。
  - **Stage 2：** 冻结主策略 $W_0$，并行训练 LoRA 适配 $BAx$（式 $h=W_0x+\frac{\alpha}{r}BAx$）；**Safety Filter** 监测姿态，超限时切换 **任务无关 Recovery Policy** $\pi_r$ 回到名义安全态 $s_{nom}$。
- **参数效率：** 可训练参数相对 FFT 约减 **99.09%**；**Actor + Critic 均需 LoRA**（仅改 Actor 会因价值函数分布偏移而不收敛）。
- **Rank 消融：** $\rho\in\{1,2,4,8\}$ 固定 75 min 预算下 **rank-1 最快恢复** 预训练性能；延长到 230 min 低秩仍占优。
- **安全：** 真机 trot/jump 微调期摔倒：FFT 无安全 14.25/69.0 次，FFT+安全 7.5/17.5，SLowRL **0.0/2.0**（4 seeds 均值）；动作变化率 $\|a_t-a_{t-1}\|$ 显著低于 FFT。
- **样本效率（sim-to-sim）：** IsaacLab→MuJoCo，匹配原性能墙钟时间 trot **-38%**、jump **-55%**。

## 核心摘录（面向 wiki 编译）

### 与相关路线的关系

| 维度 | SLowRL | 全参真机 PPO 微调 | RMA / 在线适应 | DR 零样本 |
|------|--------|-------------------|----------------|-----------|
| 更新参数量 | 极低秩 LoRA | 全网络 | 适应模块 + base | 无（部署期） |
| 安全机制 | Recovery + 滤波 | 论文对比基线可选 | 非本文焦点 | N/A |
| 平台 | Go2 动态步态 | 同类 | 多地形腿足 | 预训练即部署 |
| 核心假设 | sim2real gap ≈ **低维线性流形对齐** | 需全空间搜索 | 隐式环境变量 | 随机化覆盖 |

### 相关资料（非论文正文，工程上下文）

- **仿真 / 部署：** [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)（Isaac Lab + Go2）、[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（MuJoCo/mjlab + Go2 sim2real 链路）
- **LoRA 原论文：** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) — 本文将其用于 **高频腿足控制 PEFT**
- **安全 RL 对照：** Recovery RL、sim 训练安全 critic（论文 Related Work 讨论 sim2real gap 对安全信号的风险）

## 对 wiki 的映射

- 沉淀实体页：[SLowRL（arXiv:2603.17092）](../../wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md)
- 交叉补强：[Sim2Real](../../wiki/concepts/sim2real.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Query：Sim2Real Gap 缩减](../../wiki/queries/sim2real-gap-reduction.md)、[Comparison：Sim2Real 方法](../../wiki/comparisons/sim2real-approaches.md)、[四足机器人](../../wiki/entities/quadruped-robot.md)
