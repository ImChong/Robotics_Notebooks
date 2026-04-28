---
type: method
tags: [robustness, stability, lipschitz]
status: complete
updated: 2026-04-28
sources:
  - ../../sources/papers/lcp.md
summary: "LCP 通过对策略网络施加 Lipschitz 连续性约束，增强了物理控制器的鲁棒性和平滑度。"
---

# LCP: Lipschitz 约束策略

在物理仿真中，微小的状态扰动可能导致策略输出突变，从而造成系统失稳。**LCP** 通过限制网络的 Lipschitz 常数来解决这一问题。

## 实现方式
- 使用 **Spectral Normalization** (谱归一化) 或类似技术限制每一层神经网络的增益。
- 确保对于任意输入 , s_2$，满足 $\|\pi(s_1) - \pi(s_2)\| \le L \|s_1 - s_2\|$。

## 效果
- 显著减少了动作中的高频振荡。
- 提高了模型对外部噪声的免疫力。

## 参考来源
- [sources/papers/lcp.md](../../sources/papers/lcp.md)
