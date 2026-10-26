---
type: method
tags: [robustness, stability, lipschitz, xbpeng]
status: complete
updated: 2026-10-26
related:
  - ../entities/protomotions.md
  - ./reinforcement-learning.md
  - ../concepts/safety-filter.md
  - ../entities/mimickit.md
sources:
  - ../../sources/papers/lcp.md
summary: "LCP 通过对策略网络施加 Lipschitz 连续性约束，增强了物理控制器的鲁棒性和平滑度，防止微小扰动导致系统失稳。"
---

# LCP: Lipschitz 约束策略

**Lipschitz-Constrained Policies (LCP)** 旨在提高深度强化学习策略在物理控制任务中的数值稳定性和鲁棒性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 问题：高频不稳定性
在物理仿真中，由于接触动力学的非线性和数值误差，输入状态的微小变化可能导致网络输出动作发生剧烈跳变。这种不连续性会导致机器人关节受损或产生高频振荡。

## 核心原理：Lipschitz 约束
LCP 希望策略 $\pi$ 满足 Lipschitz 连续性，即存在常数 $L$ 使
$$
\|\pi(s_1) - \pi(s_2)\| \le L \|s_1 - s_2\|
$$
其中 $s_1, s_2$ 为任意两个观测状态，$\pi(s)$ 为策略输出的动作，$L$ 为 Lipschitz 常数。它的含义是：动作的变化幅度被限制在观测变化幅度的 $L$ 倍以内，观测上的小扰动不会被放大成动作跳变。

## 主要技术路线
| 模块 | 实现手段 | 作用 |
|------|---------|------|
| **梯度惩罚** | 在 RL 目标上加入策略输出对观测梯度范数的平方惩罚 | 以可微软约束近似 Lipschitz 约束，直接压低策略对输入的敏感度 |
| **替代平滑项** | 去掉 smoothness reward 与输出端低通滤波器 | 平滑性来自训练目标本身，避免逐机器人调多个不可微的平滑旋钮 |
| **跨平台训练** | 同一套约束用于多款人形机器人的 locomotion 训练 | 论文报告在仿真与真机上均得到平滑步态 |

## 优势
- **部署友好**：平滑性内生于策略，部署时无需额外挂低通滤波器。
- **抑噪**：限制观测噪声被放大成动作抖动；对外部推力等扰动的鲁棒性仍需结合域随机化等手段单独验证。

## 关联页面
- [protomotions](../entities/protomotions.md) — 提供大规模并行训练支持。
- [reinforcement-learning](reinforcement-learning.md) — LCP 是对标准 RL 策略的一种约束增强。
- [safety-filter](../concepts/safety-filter.md) — 类似的初衷，但 LCP 是在策略内部实现的。
- [mimickit](../entities/mimickit.md) — LCP 已集成至此研究套件。
- [LCP 论文笔记条目](../entities/paper-notebook-lcp-sim-to-real-action-smoothing.md) — arXiv:2410.11825 的 Paper Notebooks 索引实体

## 参考来源
- [sources/papers/lcp.md](../../sources/papers/lcp.md)
- Chen et al., *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies*, arXiv:2410.11825, 2024. <https://arxiv.org/abs/2410.11825>
