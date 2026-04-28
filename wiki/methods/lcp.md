---
type: method
tags: [robustness, stability, lipschitz, xbpeng]
status: complete
updated: 2026-04-28
related:
  - ./reinforcement-learning.md
  - ../concepts/safety-filter.md
  - ../entities/mimickit.md
sources:
  - ../../sources/papers/lcp.md
summary: "LCP 通过对策略网络施加 Lipschitz 连续性约束，增强了物理控制器的鲁棒性和平滑度，防止微小扰动导致系统失稳。"
---

# LCP: Lipschitz 约束策略

**Lipschitz-Constrained Policies (LCP)** 旨在提高深度强化学习策略在物理控制任务中的数值稳定性和鲁棒性。

## 问题：高频不稳定性
在物理仿真中，由于接触动力学的非线性和数值误差，输入状态的微小变化可能导致网络输出动作发生剧烈跳变。这种不连续性会导致机器人关节受损或产生高频振荡。

## 核心原理：Lipschitz 约束
LCP 通过限制神经网络的 Lipschitz 常数 $，确保：
290293\|\pi(s_1) - \pi(s_2)\| \le L \|s_1 - s_2\|290293
这意味着策略的输出变化率被严格限制在输入变化率的比例范围内。

## 主要技术路线
| 模块 | 实现手段 | 作用 |
|------|---------|------|
| **权重归一化** | [域随机化](../concepts/domain-randomization.md) Spectral Normalization | 限制网络层的最大奇异值，保证层间 Lipschitz 连续 |
| **残差连接** | Residual Mapping | 通过残差结构平衡网络的表达能力与约束强度 |
| **Lipschitz 估计** | Power Iteration | 实时在线估计网络的 Lipschitz 常数，用于自适应调节 |

## 优势
- **部署友好**：生成的动作序列极其平滑，适合在真机硬件上直接执行。
- **抗扰动**：对传感器噪声和外部推力具有天然的抑制作用。

## 关联页面
- [[reinforcement-learning]] — LCP 是对标准 RL 策略的一种约束增强。
- [[safety-filter]] — 类似的初衷，但 LCP 是在策略内部实现的。
- [[mimickit]] — LCP 已集成至此研究套件。

## 参考来源
- [sources/papers/lcp.md](../../sources/papers/lcp.md)
- Peng et al., *Lipschitz-Constrained Policies for Physics-Based Character Control*, SIGGRAPH 2023.
