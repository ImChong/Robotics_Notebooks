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
- **通读：** 已完成
- **核对范围：** `2603.17092v1` 全文 §I–§VIII（含 Fig. 1–9、Table I–II 数值及正文 §V-B 细微数值差异核对）
- **官方代码：** 论文未给出官方代码仓库链接
- **一句话说明：** 冻结仿真预训练策略，仅用 **LoRA 低秩适配 + 训练期 Recovery Policy 安全滤波**，在真机上安全高效微调动态运动策略；rank-1 即可恢复性能，相对全参 PPO 微调约 **46.5%** 墙钟时间缩短、近零摔倒。

## 摘要级要点

- **问题：** 运动策略 sim-to-real 后性能退化；直接在硬件上做 **全参 fine-tuning（FFT）** 样本效率低且易机械损伤。
- **方法 SLowRL：**
  - **Stage 1：** 高保真仿真 IsaacLab（PhysX）预训练（contact-explicit 架构，期望接触点/时序条件化）。
  - **Stage 2：** 冻结主策略全层权重 $W_0$，在 Actor 与 Critic **所有线性层**（ELU 激活之前）并行注入低秩 LoRA 适配 $h = W_0 x + \frac{\alpha}{r}BAx$（初始化 $A\sim\mathcal{N}(0,\sigma^2),\ B=0$ 保证起点等于预训练行为；Actor 末层另加可训练探索噪声参数）；相比 FFT 减少约 **99.09%** 可训练参数（LoRA LR $10^{-2}$ vs FFT $10^{-3}$）。
  - **Safety Filter + Recovery Policy：** 离散安全触发器每步检测本体姿态（pitch/roll 超限），超限时硬切换并覆盖主策略动作，调用强域随机化训练的**任务无关 Recovery Policy** $\pi_r$ 拉回名义直立低速安全态 $s_{\text{nom}}$。
- **核心发现与消融：**
  - **Actor + Critic 同步 LoRA 必要性：** 仅适配 Actor 时 Critic 仍锚定源仿真分布，价值损失停在 **0.2** 且不收敛；Actor + Critic 全层适配可将价值损失降至 **0.007**（2 个数量级差距），且优于仅改输入/输出层或末层。
  - **Rank 消融：** $\rho\in\{1,2,4,8\}$ 固定 75 min 预算下 **rank-1 ($\rho=1$) 最快恢复** 预训练性能；延长到 230 min 低秩仍占优，高秩反而放大接触不连续下的梯度噪声。
  - **安全与平滑（Sim-to-Sim, IsaacSim → MuJoCo, 4 seeds）：** Table I 报告平均摔倒次数 Trot/Jump：FFT 无安全 `14.25 / 69.0`，FFT+安全 `7.5 / 17.5`（注：正文 §V-B 行文笔误写为 `x` 与 `27`，此处以 Table I 为准并保留原文差异说明），SLowRL 为 **`0.0 / 2.0`**；Table II 报告动作变化率 $\|a_t-a_{t-1}\|$ 下降 Trot **`88.9%`**（FFT `38.9%`）、Jump **`35.7%`**（FFT `5.7%`）。
  - **样本效率与真机验证：** IsaacLab → MuJoCo 匹配原性能墙钟时间 Trot **-38%**、Jump **-55%**（均值 **46.5%**）；Unitree Go2 真机（Vicon 动捕 + RTX 3080 Ti 工作站，60 min 跳跃训练）SLowRL 将平均回报从 **`-19` 提升至 `-7`**（FFT 平台在 **`-15`**），接触模式一致性提升 **`+9%`**。
  - **原文声明局限（§VII）：** 当前仅验证于平地与中等扰动，复杂非结构化地形可能需要 $\rho>1$；Recovery 安全集 $s_{\text{nom}}$ 仍需人工定义；基于 on-policy PPO 无法复用历史真机轨迹，作者规划探索 off-policy/hybrid 微调。

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
