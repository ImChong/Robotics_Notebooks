---
type: method
tags: [il, behavior-cloning, supervised-learning, manipulation, covariate-shift]
status: complete
updated: 2026-09-05
summary: "Behavior Cloning 把专家演示转成监督学习问题，是机器人模仿学习最简单也最常用的基线。"
related:
  - ./imitation-learning.md
  - ../concepts/rl-runner.md
  - ./dagger.md
  - ./inverse-reinforcement-learning.md
  - ./diffusion-policy.md
  - ./action-chunking.md
  - ../comparisons/rl-vs-il.md
  - ../tasks/manipulation.md
  - ../entities/kinetiq-ascend.md
  - ../entities/paper-why-action-chunking-improves-bc.md
  - ../entities/paper-nestdex.md
  - ../entities/paper-spd.md
  - ../concepts/behavioral-cloning-mysteries.md
  - ../entities/paper-revisiting-open-loop-action-chunking.md
sources:
  - ../../sources/personal/rl_runner_types.md
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/why_action_chunking_improves_bc_corl2026.md
  - ../../sources/papers/nestdex_arxiv_2608_13362.md
  - ../../sources/blogs/seohong_behavioral_cloning_mystery.md
---

# Behavior Cloning（行为克隆）

**Behavior Cloning, BC**：把专家演示数据当作监督学习数据集，直接学习从观测到动作的映射，是模仿学习最直接的做法。

## 一句话定义

给机器人一堆“专家在这个状态下应该怎么做”的样本，训练一个策略去直接复现这些动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BC | Behavior Cloning | 将状态映射到动作的监督式模仿，易受分布偏移影响 |
| DAgger | Dataset Aggregation | 迭代收集策略诱导状态下的专家标注以纠偏的模仿学习方法 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构，常与 ALOHA 配套 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

- 它几乎是所有模仿学习 pipeline 的起点：先训一个能跑的 BC baseline，再谈 DAgger、Diffusion Policy 或 IL+RL。
- 在奖励函数很难设计、但演示数据容易拿到的任务里，BC 往往是最低门槛方案。
- 许多真机操作系统都会先用 BC 做 warm start，再用更复杂方法提升鲁棒性。
- **工业操作的新共识（2026）：** BC 往往只需覆盖 **行为模态**；**速度与近完美可靠性** 需 RL 在真实动力学下优化——见 [KinetIQ Ascend](../entities/kinetiq-ascend.md) 对 **示教速度上限、因果混淆、失败代价不可见** 的讨论。
- **真机风格数据会改写「标准 BC 直觉」：** 过拟合有时更好、开环 chunk 优于逐步闭环、简单状态任务也要极大网络、无限数据下特征缩放仍改成功率——见 [BC Mysteries](../concepts/behavioral-cloning-mysteries.md)（仿真复现，基准尚未开源）。

## 输入、输出与训练目标

给定专家数据集 $D = \{(o_i, a_i)\}$，BC 通常优化：

$$
\min_\theta \; \mathbb{E}_{(o,a)\sim D}[\ell(\pi_\theta(o), a)]
$$

常见设定：
- **输入**：图像、关节状态、末端位姿、历史动作等观测
- **输出**：关节目标、末端动作、action chunk 或离散动作 token
- **损失**：MSE、L1、交叉熵、负对数似然

## 核心优点

### 1. 简单直接
它不需要环境交互、不需要在线探索，也不需要 reward engineering。

### 2. 数据效率高于纯 RL
在固定专家数据上训练，通常比从零探索的 RL 更快进入“能做事”的区间。

### 3. 工程上容易落地
训练和部署都像标准 supervised learning，适合先做 baseline、集成到已有感知模型、或作为大模型动作头。

## 核心局限

### 1. Covariate Shift / Distribution Shift
训练时看到的是专家访问到的状态，部署时策略一旦出错，就会进入训练集中没见过的状态分布。

BC 的监督目标只在专家诱导的状态分布 \(d_{\pi^*}\) 上取期望，闭环部署却是在策略自己诱导的 \(d_{\pi_\theta}\) 上被评测——**训练分布与评测分布不是同一个**，这是纯 BC 与普通监督学习最本质的差别，也是 covariate shift（分布偏移）一词在模仿学习里的确切所指。Ross & Bagnell 给出的经典刻画是：单步误差为 \(\epsilon\) 的 BC 在 horizon \(H\) 上最坏可退化到 \(\mathcal{O}(\epsilon H^2)\)，而持续在策略诱导分布上补标注的 [DAgger](./dagger.md) 可做到 \(\mathcal{O}(\epsilon H)\)——**多出来的那个 \(H\) 就是分布偏移的代价**。

工程上有三条直接推论：

- **验证集选错分布就测不出问题**：专家分布上的 NLL / MSE 再低，也不保证 \(d_{\pi_\theta}\) 上不崩；应改测策略诱导状态下的动作误差（与下方「Train 指标与闭环成功率不对齐」同源）。
- **数据要覆盖「偏离之后怎么回来」**：只演示完美轨迹时，\(d_{\pi_\theta}\) 的尾部状态完全没有监督信号；[DA-Nav](../entities/paper-da-nav.md) 的 recovery 消融给出了定量对照。
- **采集时主动注入扰动**：在演示过程中加噪声（DART 等做法）相当于让专家顺带标注邻域状态，是不做在线交互时最省事的近似。

### 2. Compounding Error
单步小误差会沿着闭环执行不断累积，序列越长、任务越长 horizon，问题越明显。BC 并不是“每步都独立无害”的方法。[Why Action Chunking Improves BC](../entities/paper-why-action-chunking-improves-bc.md) 进一步给出尺度：Markov BC 可遭 \(\Omega(2^H\epsilon)\) 下界，而 **delayed policy**（\(a_t\mid o_{t-n}\)）与 action chunking 共享 \(\mathcal{O}((k+1)^{H/k}\epsilon)\) 上界——缓解复合误差不一定要「播放整段动作块」。

### 3. 受限于专家上界
如果数据里没有恢复动作、异常姿态或罕见接触，BC 通常也学不会这些行为。[DA-Nav](../entities/paper-da-nav.md) 在户外 VLN 消融中给出定量对照：去掉 recovery 轨迹后 CSR 从约 **98%** 掉到 **15%**，说明「只仿完美专家」对闭环纠偏不足。

### 4. Train 指标与闭环成功率不对齐（真机风格数据）
在窄分布、时间强相关的人类风格演示上，验证 **flow / BC 损失恶化** 时成功率仍可能上升；同分布更大数据集有时更差。更相关的代理是策略诱导状态下的动作误差，而不是专家分布上的 NLL。系统整理见 [BC Mysteries](../concepts/behavioral-cloning-mysteries.md)。

## 主要技术路线

| 问题 | 常见缓解 |
|------|---------|
| 分布漂移 | DAgger 在线聚合策略访问到的新状态 |
| 长时序误差 | Action Chunking、序列模型、闭环再规划 |
| 多模态动作 | Diffusion Policy、Flow Matching、Mixture Density |
| 真机鲁棒性不足 | 数据增强、传感器噪声注入、真实数据微调 |

## 与 DAgger、Diffusion Policy 的关系

- **BC**：最简单，离线监督学习基线。
- **DAgger**：仍然学监督映射，但会反复收集“当前策略真正会访问到的状态”，核心是修复 covariate shift。
- **Diffusion Policy / π₀**：仍可看作 BC 范式的生成式升级，重点解决多模态动作和长时序建模。

## 在机器人里的典型应用

### 操作
- 桌面抓取、装配、双手协作
- 遥操作数据蒸馏为离线策略
- VLA / Foundation Policy 的动作头微调基线

### 移动与 locomotion
- 用 MoCap、教师策略或 privileged teacher 生成数据，再做学生策略蒸馏
- 常作为 IL+RL 混合流程的第一步，而不是最终控制器

## 常见误区

- **误区 1：BC 与 DAgger 等价。**
  不是。DAgger 的关键价值正是在于持续覆盖策略部署分布，通常比纯 BC 更能处理分布漂移。
- **误区 2：BC 的累积误差和序列长度无关。**
  错。horizon 越长，早期偏差越容易滚雪球。
- **误区 3：只要模型够大，BC 就天然鲁棒。**
  模型容量能帮助拟合，但不能替代分布覆盖。反过来说：真机风格数据上 **容量不够** 会让最简单的 pick-and-place 也学不会——「任务简单所以小 MLP 够了」同样是误区。
- **误区 4：闭环逐步执行一定优于开环 chunk。**
  纯 \(\pi(a_t\mid s_t)\) 在无限数据 BC 上可以完全失败；开环是在补偿非马尔可夫演示。加长历史也不自动等于修好——见 [BC Mysteries](../concepts/behavioral-cloning-mysteries.md) 与 [Revisiting Open-Loop](../entities/paper-revisiting-open-loop-action-chunking.md) 的对读。

## 参考来源

- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — DAgger / BC / ACT / Diffusion Policy 的 ingest 档案
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 生成式模仿学习如何扩展传统 BC
- [sources/papers/why_action_chunking_improves_bc_corl2026.md](../../sources/papers/why_action_chunking_improves_bc_corl2026.md) — chunk / delay 对 BC 复合误差与部署协议的机制分析
- [SPD 论文归档](../../sources/papers/spd_corl_2026.md) — 仿真预训练后的真机 BC 微调对照
- [Behavioral cloning mystery（Seohong Park，2026-08）](../../sources/blogs/seohong_behavioral_cloning_mystery.md) — 真机风格数据上的四条可复现反直觉
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* — 解释为什么纯 BC 会受到 covariate shift 影响

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [RL Runner（训练循环编排）](../concepts/rl-runner.md) — Imitation Runner 的离线监督形态：读演示、不探索
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md) — BC 在不同动作空间下的数学形式化定义
- [DAgger](./dagger.md)
- [Inverse Reinforcement Learning](./inverse-reinforcement-learning.md) — 不直接拟合动作，先推断 $r$ 再优化；覆盖不足时仍会外推失败
- [Diffusion Policy](./diffusion-policy.md)
- [Manipulation](../tasks/manipulation.md)
- [RL vs Imitation Learning](../comparisons/rl-vs-il.md)
- [KinetIQ Ascend](../entities/kinetiq-ascend.md) — BC 预训练 + 真机 PPO 突破工业可靠性/速度天花板
- [DA-Nav](../entities/paper-da-nav.md) — 户外 VLN：专家+recovery 数据相对纯 BC 的纠偏消融
- [Emergent Transfer](../entities/paper-emergent-transfer-cross-config.md) — 跨配置 BC 共训中遗留数据的三相有效性
- [Action Chunking](./action-chunking.md) — 对单步 BC 的时间窗扩展；部署协议可与训练目标解耦
- [Why Action Chunking Improves BC](../entities/paper-why-action-chunking-improves-bc.md) — Delay / RDE：何时不必真的执行 chunk
- [NestDex](../entities/paper-nestdex.md) — 外层 BC 用 H-VAE 手 latent；示范来自 copilot 而非全 DoF 遥操作（arXiv:2608.13362）
- [SPD](../entities/paper-spd.md) — 仿真预训练后的真机 BC 微调，五项任务均胜过从零（CoRL 2026）
- [BC Mysteries](../concepts/behavioral-cloning-mysteries.md) — 过拟合 / 开环 / 大模型 / 特征缩放四条真机风格现象
- [Revisiting Open-Loop Execution](../entities/paper-revisiting-open-loop-action-chunking.md) — 加长观测上下文后闭环可赢开环

## 推荐继续阅读

- Ross et al., *DAgger* — 经典交互式 IL 方法
- Zhao et al., *ACT* — 用 action chunking 缓解长时序误差
- Chi et al., *Diffusion Policy* — 生成式方法如何超越传统 BC
- Lazzati et al., [*Why Does Action Chunking Improve BC?*](https://action-chunking.github.io/) — 机制消融与 RDE 部署
