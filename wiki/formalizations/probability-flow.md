---
type: formalization
tags: [math, probability, generative-ai, flow-matching, diffusion]
status: complete
updated: 2026-07-04
related:
  - ../methods/hy-motion-1.md
  - ../methods/diffusion-motion-generation.md
sources:
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/hy_motion_arxiv_2512_23464.md
summary: "概率流（Probability Flow）提供了一个统一的数学框架来描述扩散模型与流匹配，通过将离散的去噪步转化为连续的可微常微分方程（ODE），实现了生成过程的极致加速与可控性。"
---

# Probability Flow (概率流形式化)

在具身智能的生成式动作建模（如 **π₀** 或 **Diffusion Policy**）中，**概率流 (Probability Flow)** 是连接噪声分布与真实动作分布的数学“传送带”。它将生成过程描述为一种连续的动力学系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| DiT | Diffusion Transformer | 以 Transformer 为骨干的扩散生成架构 |
| SMPL | Skinned Multi-Person Linear Model | 常见人体参数化模型与重定向源 |

## 数学定义：ODE 视角

给定从简单噪声分布 $p_0$ 到复杂动作分布 $p_1$ 的变换。概率流通过一个常微分方程 (ODE) 描述样本 $x_t$ 随时间 $t \in [0, 1]$ 的演化：

$$ dx_t = v(x_t, t) dt $$

其中 $v(x, t)$ 是 **速度场 (Velocity Field)**。

### 1. 与扩散模型 (Diffusion) 的关系
在扩散模型中，概率流 ODE (PF-ODE) 能够以确定性的方式复现随机微分方程 (SDE) 的边际分布。通过得分匹配 (Score Matching)，速度场可表示为：
$$ v(x_t, t) = f(x_t, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x_t) $$
其中 $\nabla_x \log p_t(x_t)$ 即为得分函数 (Score Function)。

### 2. 流匹配 (Flow Matching, FM)
流匹配是 π₀ 等最新 VLA 模型采用的技术。它不再通过复杂的扩散/去噪链条，而是直接回归一个最优的速度场。
- **目标函数**：直接最小化预测速度与理想线性插值速度之间的差异：
  $$ \mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, x_0, x_1} [ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 ] $$

### 3. 十亿级 DiT + 流匹配（HY-Motion 1.0）

[HY-Motion 1.0](../methods/hy-motion-1.md)（arXiv:2512.23464）把 **DiT + Flow Matching** 推到十亿参数级文本驱动人体运动生成；数据管线经 GVHMR → SMPL-H 重定向得到 **>3000 小时** 语料，是 **概率流 ODE 在人体运动域** 的可复现锚点。

## 为什么对机器人重要

1. **推理速度**：传统的扩散模型需要 50-100 步迭代；概率流 ODE 可以通过高级求解器（如 RK45）在 1-3 步内生成动作，满足机器人的实时性需求。
2. **时域平滑性**：由于动作生成被建模为连续流，输出的轨迹序列 $a_{t:t+k}$ 在数学上具有更好的 C¹ 连续性，减少了电机的冲击。
3. **少样本适应**：通过微调（Fine-tuning）速度场，模型可以极快地学习新环境下的动作偏好。

## 关联页面
- [π₀ (Pi-zero) 策略模型](../methods/π0-policy.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源
- Lipman, Y., et al. (2022). *Flow Matching for Scalable Simulation*.
- Song, Y., et al. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*.
- [sources/papers/hy_motion_arxiv_2512_23464.md](../../sources/papers/hy_motion_arxiv_2512_23464.md) — HY-Motion 1.0
