# SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion（arXiv:2603.17092）

> 论文来源归档（ingest）

- **标题：** SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion
- **类型：** paper / quadruped locomotion / sim2real / online fine-tuning / PEFT
- **arXiv abs：** <https://arxiv.org/abs/2603.17092>
- **arXiv HTML：** <https://arxiv.org/html/2603.17092v1>
- **PDF：** <https://arxiv.org/pdf/2603.17092>
- **机构：** McGill University（Hsiu-Chin Lin 等）、Mila – Quebec AI Institute / Université de Montréal（Glen Berseth）、Mila（Majid Khadiv）、其他合作者（Elham Daneshmand、Shafeef Omar）
- **硬件：** Unitree Go2 四足；真机实验用 Vicon 提供基座位姿与速度真值
- **仿真栈：** Isaac Lab / IsaacSim（PhysX）预训练 → MuJoCo 目标域 sim-to-sim；真机 on-policy PPO 微调
- **公开代码：** 截至入库日 arXiv 页未挂官方实现仓库（与同名 NLP 项目 `qlabs-eng/slowrun` 无关）
- **入库日期：** 2026-05-25
- **一句话说明：** 冻结仿真预训练策略，在 Actor/Critic 各层并行注入 **LoRA（默认 rank=1）** 做真机/跨仿真微调，并用 **任务无关 recovery policy + 姿态安全滤波** 在训练期兜底，相对全参 PPO 微调报告约 **46.5%** 墙钟时间缩短与近零跌倒。

## 摘要级要点

- **问题：** 运动策略 sim-to-real 后性能常退化；直接在硬件上做 **全参（FFT）PPO 微调** 样本效率低且易机械损伤（跌倒、关节限位、bang-bang 高频动作）。
- **核心假设：** 腿足 sim-to-real 对齐本质是 **低维策略修正**——仿真已学到主要运动技能，现实缺口可用 **极低秩权重扰动** 吸收；实证 **ρ=1** 往往最快恢复预训练回报。
- **SLowRL 三件套：**
  1. **冻结主策略** $W_0$（Isaac Lab 接触显式架构 [13] 预训练：足端期望接触目标/时序条件化）。
  2. **LoRA 适配器：** 每层 $h=W_0 x + \frac{\alpha}{r}BAx$，$A\sim\mathcal{N}(0,\sigma^2)$、$B=0$ 初始化使初始行为等同预训练；可训练参数量约为 FFT 的 **0.91%**（论文称约 **99.09%** 减少）。
  3. **Recovery policy $\pi_r$ + Safety Filter：** 倾角/姿态超限时用保守恢复动作覆盖主策略输出；$\pi_r$ 任务无关、DR 训练，将广分布状态拉回名义安全态 $s_{nom}$（直立、低速度、仅本体感知定义）。
- **训练细节：** LoRA-PPO；LoRA 学习率 $10^{-2}$，FFT $10^{-3}$；**Actor-only LoRA 失败**，须 **Actor+Critic 同步 LoRA** 以重对齐价值函数（源/目标动力学分布偏移）。
- **任务：** Go2 **trot**（周期稳定）与 **jump**（高动态冲量）；对比 Zero-Shot、FFT（无/有安全机制）、SLowRL。
- **安全（4 seed 平均跌倒次数）：** Trot：FFT 无安全 14.25、有安全 7.5、SLowRL **0**；Jump：69 / 17.5 / **2**。动作变化率 $\|a_t-a_{t-1}\|$ 降幅 trot **88.9%** vs FFT **38.9%**。
- **样本效率（sim-to-sim，墙钟至恢复 Isaac 预训练回报）：** Trot **38%** 时间、Jump **55%** 时间；真机 60 min 内优于 FFT 叙事。
- **Rank 消融：** $\rho\in\{1,2,4,8\}$ 固定 75 min 预算，**ρ=1** 最快达预训练水平；延长到 230 min 低秩仍占优。

## 核心摘录（面向 wiki 编译）

### 与 DR / SysID / RMA 的边界

| 路线 | 何时用 | 与 SLowRL 关系 |
|------|--------|----------------|
| Domain Randomization | 预训练期鲁棒化 | SLowRL 假设 **已有** 仿真策略，在 **部署后** 做短微调 |
| 全参真机 RL 微调 | 算力/安全预算充足 | SLowRL 用 LoRA + recovery **降维 + 兜底** |
| Recovery RL [21] | 安全 critic/约束 | 本文 recovery **独立策略**，非仅 critic；且与 LoRA 正交增益 |

### 与 Any2Any / 人形 LoRA 用法的对照（索引级）

- **SLowRL：** **同平台**（Go2）跨仿真/真机 **动力学对齐**；rank-1、真机安全微调。
- **Any2Any（arXiv:2605.23733）：** **跨人形 embodiment** 的 WBT 迁移；先 **运动学对齐** 再 **动力学 LoRA**——见 [`any2any_arxiv_2605_23733.md`](any2any_arxiv_2605_23733.md)。

## 对 wiki 的映射

- 沉淀实体页：[SLowRL 安全 LoRA 真机微调（arXiv:2603.17092）](../../wiki/entities/paper-slowrl-safe-lora-locomotion.md)
- 交叉补强：[Sim2Real](../../wiki/concepts/sim2real.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[四足机器人](../../wiki/entities/quadruped-robot.md)、[Unitree](../../wiki/entities/unitree.md)、[Balance Recovery](../../wiki/tasks/balance-recovery.md)
