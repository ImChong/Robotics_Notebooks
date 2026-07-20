---
type: method
tags: [il, diffusion, manipulation, generative-model]
status: complete
summary: "Diffusion Policy 用多步去噪生成动作序列，擅长处理多模态和长时序的机器人操作行为。"
updated: 2026-07-20
---

# Diffusion Policy

**Diffusion Policy**：将扩散生成模型（Diffusion Model）用于机器人模仿学习，通过逆扩散过程从噪声中生成动作序列的策略学习方法。

## 一句话定义

把"生成图片"的扩散模型换成"生成动作"——**不是直接预测下一个动作，而是通过多步去噪过程，从高斯噪声中逐步生成一个动作序列。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DP | Diffusion Policy | 用扩散模型生成动作序列的 IL 方法 |
| IL | Imitation Learning | 从示教学习，DP 是其生成式分支 |
| BC | Behavior Cloning | 确定性回归对照基线 |
| ACT | Action Chunking Transformer | 常与 DP 并列的序列动作预测架构 |
| Sim2Real | Simulation to Real | 操作策略迁移真机的后续阶段 |

## 为什么重要

传统模仿学习（BC）的核心问题是**多模态动作分布**：
- 同一个状态下，专家可能有多种合理做法（比如绕左边或绕右边）
- BC 用 MSE 或 NLL 直接回归时，容易学出各种动作的"平均"，变成没有意义的中间动作
- Diffusion Policy 通过扩散过程，天然支持多模态分布的表达

这使得它在操作类任务中大幅超越了传统 BC：
- 更鲁棒的接触处理
- 更灵活的技能组合
- 在 long-horizon 操作任务中保持高成功率

## 主要技术路线

### 扩散过程（Diffusion Process）

扩散模型分两个过程：

**正向过程（加噪声）：**

$$q(x_k | x_{k-1}) = \mathcal{N}(x_k; \sqrt{1-\beta_k} x_{k-1}, \beta_k I)$$

从真实动作 $x_0$ 出发，逐步加噪声，到 $x_K \sim \mathcal{N}(0, I)$（纯噪声）。

**逆向过程（去噪声，即推理时）：**

$$p_\theta(x_{k-1} | x_k, s) = \mathcal{N}(x_{k-1}; \mu_\theta(x_k, k, s), \Sigma_k)$$

从纯噪声 $x_K$ 出发，通过神经网络预测噪声，逐步去噪，得到动作 $x_0$。

**$s$** 是当前观测（图像、关节状态、位姿等），作为条件输入。

### 为什么能处理多模态

不同于 BC 直接预测均值，扩散过程是一个**概率生成过程**，可以从同一个状态条件下采样到多种合理动作——自然地覆盖多模态分布。

### 动作块（Action Chunk）预测

Diffusion Policy 通常预测一段动作序列（Action Chunk），而不是单步动作：
- 减少高频动作的抖动
- 使策略能做更长时间的协调规划
- 典型长度：16～32 步

## 两种主要实现变体

### DDPM（基于 UNet 的扩散策略）
- 噪声预测器使用 UNet 架构
- 支持图像观测（视觉策略）
- 推理步数多（DDPM 通常 100 步），较慢
- 适合离线桌面操作任务

### DDIM + Transformer 变体
- 改用 Transformer 做噪声预测
- 支持 DDIM 加速（10 步以内可完成推理）
- 速度更快，适合实时控制

## 和传统 BC 的对比

| | Behavior Cloning | Diffusion Policy |
|--|---------|---------|
| 多模态 | ❌ 均值化，无法表达 | ✅ 天然支持 |
| 接触丰富任务 | ❌ 容易失败 | ✅ 显著更好 |
| 训练速度 | 快 | 较慢（需要扩散步骤） |
| 推理速度 | 实时 | 需要加速变体（DDIM） |
| 实现复杂度 | 简单 | 较高 |

## 在机器人中的应用场景

### 1. 桌面操作（Tabletop Manipulation）
- 抓取、放置、折叠、插入等精细操作
- 论文中常见 benchmark：Push-T、Robomimic、RoboAgent

### 2. 双手操作 / 全身操作
- 人形机器人上肢操作任务
- 配合力控或阻抗控制的混合策略

### 3. 技能组合（Skill Composition）
- 多技能 diffusion 模型的条件切换
- 配合语言或任务描述做 conditioning

## 常见挑战

### 推理速度
标准 DDPM 推理需要 100 步，在 50Hz 控制频率下不可行。
解决方案：DDIM（10 步以内）、Consistency Models、Flow Matching。

**免训练缓存加速（Muninn，RSS 2026）**：[Muninn](../entities/paper-muninn-trajectory-diffusion-acceleration.md) 把逐步 denoiser 当作可预算复用：用廉价 **probe** + 采样器 **解析灵敏度** 经 conformal 标定，在可证轨迹偏离界下跳过冗余 forward；论文在 RLBench / Meta-World / DP3 上报告约 **40%** 推理延迟下降且成功率接近 Full，且可与蒸馏少步模型叠加。

### Sim2Real 挑战
扩散策略对观测分布非常敏感，从仿真迁移到真实时视觉 gap 尤为明显。
常见做法：真实数据微调、领域随机化、sim2real 感知适配。

### 数据需求
比 BC 需要更多、更高质量的演示数据，数据多样性对多模态表达至关重要。

## 参考来源

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023) — Diffusion Policy 原论文
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, 2023) — 操作任务 IL 代表（动作块预测）
- [Diffusion Policy 项目主页](https://diffusion-policy.cs.columbia.edu/)
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — Chi 2023 / π₀ / BESO / ACT / Consistency Policy
- **ingest 档案：** [sources/papers/defi_arxiv_2604_16391.md](../../sources/papers/defi_arxiv_2604_16391.md) — DeFI：GFDM 前向 + GIDM 逆动力学解耦预训练后再用扩散适配器端到端耦合微调
- **ingest 档案：** [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI：全身 47-D 稀疏关键点 + DINOv2 条件的扩散高层（arXiv:2605.03452）
- **ingest 档案：** [sources/papers/muninn_arxiv_2605_09999.md](../../sources/papers/muninn_arxiv_2605_09999.md) — Muninn：免训练轨迹扩散缓存包装器，visuomotor DP 推理加速与可证偏差预算（RSS 2026）

## 关联页面

- [扩散模型（概念）](../concepts/diffusion-model.md) — 通用前向加噪/逆向降噪机制与架构演进
- [Diffusion-based Motion Generation](./diffusion-motion-generation.md) — 扩散模型在移动任务中的应用
- [Imitation Learning](./imitation-learning.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Policy Optimization](./policy-optimization.md)
- [Manipulation](../tasks/manipulation.md)
- [Sim2Real](../concepts/sim2real.md)
- [BifrostUMI（论文实体）](../entities/paper-bifrost-umi.md) — 人形全身 visuomotor 的扩散高层实例
- [Muninn（论文实体）](../entities/paper-muninn-trajectory-diffusion-acceleration.md) — 扩散策略/轨迹扩散的免训练推理加速与 conformal 偏差证书
- [OAT 有序动作 Tokenization](../entities/paper-oat-ordered-action-tokenization.md)


## 推荐继续阅读

- [机器人论文阅读笔记：iDP3 Generalizable Humanoid Manipulation with 3D Diffusion Policies](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/iDP3_Generalizable_Humanoid_Manipulation_with_3D_Diffusion_Policies/iDP3_Generalizable_Humanoid_Manipulation_with_3D_Diffusion_Policies.html)
- [机器人论文阅读笔记：Diffusion Policy](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/Diffusion_Policy/Diffusion_Policy.html)
- Chi et al., [*Diffusion Policy*](https://arxiv.org/abs/2303.04137) — 原论文
- Zhao et al., [*ACT: Action Chunking with Transformers*](https://arxiv.org/abs/2304.13705) — 动作块预测方法
- Black et al., [*π0: A Vision-Language-Action Flow Model for General Robot Control*](https://www.physicalintelligence.company/blog/pi0) — flow matching 路线

## 一句话记忆

> Diffusion Policy 把扩散生成模型引入模仿学习，天然解决了 BC 无法表达多模态动作分布的问题，是当前操作类任务中最强的 IL 方法之一。
