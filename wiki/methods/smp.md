---
type: method
tags: [score-matching, diffusion, generative-prior, humanoid, unitree-g1]
status: complete
updated: 2026-04-28
related:
  - ./amp-reward.md
  - ./ase.md
  - ../entities/unitree-g1.md
sources:
  - ../../sources/papers/smp.md
summary: "SMP (Score-Matching Motion Priors) 通过预训练扩散模型作为“冻结奖励器”，实现了高效、可组合且无需原始数据的运动模仿学习。"
---

# SMP: 基于得分匹配的可复用运动先验

**SMP** 代表了从对抗模仿学习（如 [[amp-reward]]）向生成式先验引导学习的范式演进。它将复杂的运动分布建模为一个连续的得分场（Score Field），并以此指导 RL 策略。

## 核心技术路线

### 1. 冻结的扩散模型作为奖励
与 AMP 不同，SMP 不需要判别器与策略共同训练。
- **预训练**：在动作数据集上预训练一个扩散模型。
- **冻结奖励**：在 RL 阶段，扩散模型被冻结，不再需要原始数据集。这使得模型极易于在不同任务和环境中复用。

### 2. SDS (Score Distillation Sampling)
SMP 借鉴了文本转图像领域的 SDS 技术：
- 策略生成的动作片段被添加噪声，然后输入扩散模型。
- 扩散模型预测噪声，其预测值与实际添加噪声的差异（即得分方向）被转化为奖励：
  284016r_{SMP} = -\mathbb{E}_{t, \epsilon} [ \| \epsilon - \epsilon_\theta(x_t; t) \|^2 ]284016
- **直观理解**：奖励函数在告诉策略：“如果你能让动作更接近扩散模型认为的‘无噪声自然动作’，你就会得到更高分。”

### 3. ESM (Ensemble Score-Matching)
为了解决扩散模型在不同噪声水平（Timesteps）下输出不一致导致的奖励震荡，SMP 引入了 **ESM**：
- 在多个时间步 $ 上并行评估并取平均值。
- 这提供了更平滑、方差更低的梯度信号，使得物理模拟器中的训练更加稳定。

### 4. GSI (Generative State Initialization)
传统的 RL 需要从数据集里随机采样初始状态（Reference State Initialization, RSI）。SMP 通过 **GSI** 实现了自给自足：
- 直接用扩散模型生成合法的、处于运动中的初始位姿和速度。
- 实验证明 GSI 能够完全替代 RSI，且能覆盖更多样化的状态空间。

## 风格组合 (Style Composition)
由于 SMP 是基于得分的，可以通过简单的线性加权来组合不同的风格先验：
284016\nabla \log p_{mix}(x) \approx w_1 \nabla \log p_1(x) + w_2 \nabla \log p_2(x)284016
这使得机器人可以在没有见过“跑步时挥手”数据的情况下，通过“跑步”和“挥手”两个独立先验学会该动作。

## 硬件实现：Unitree G1
SMP 在 **[[unitree-g1]]** 人形机器人上完成了真机验证。
- **Sim2Real 鲁棒性**：得益于扩散模型捕获的结构化分布，生成的策略在面对传感器噪声和外部推力时表现出极强的恢复能力。
- **应用场景**：自然行走、平衡恢复、以及跨地形移动。

## 比较：SMP vs AMP
| 特性 | [[amp-reward]] | [[smp]] |
|------|-----------|---------|
| **训练模式** | 对抗学习 (需要判别器) | 生成式先验 (冻结奖励器) |
| **数据依赖** | 训练时必须实时读取轨迹数据 | 仅在预训练时需要，RL 时无需数据 |
| **稳定性** | 容易发生模式塌陷 (Mode Collapse) | 极高，受益于扩散模型的平滑性 |
| **组合性** | 难 | 易 (梯度场线性叠加) |

## 参考来源
- [sources/papers/smp.md](../../sources/papers/smp.md)
- Mu et al., *SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control*, 2026.
