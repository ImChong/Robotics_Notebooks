---
type: roadmap_page
tags: [roadmap, locomotion, humanoid, rl, sim2real]
status: active
summary: "面向人形机器人运动控制算法工程师的四阶段学习路线：建模控制 → 仿真训练 → Sim2Real → 进阶专题。"
updated: "2026-04-20"
---

# Humanoid Control Roadmap

> 本页是 [[wiki/roadmaps/humanoid-control-roadmap]] 的简短版摘要。
> **完整版路线请看 → [[roadmap/route-a-motion-control]]**

面向人形机器人运动控制算法工程师的学习研究路线。

## 适合谁

- 机械/控制/机器人背景，想转人形机器人运控
- 有一定编程基础（Python, C++）
- 了解基本线性代数和控制理论更好（但不是必须）

## 先修知识

### 核心必学
1. **Python 基础**：能读懂和改代码
2. **机器人学基础**：正逆运动学、正逆动力学概念（不需要特别深）
3. **强化学习基础**：理解 MDP、policy、value function 概念

### 推荐资源
- [斯坦福《机器人学导论》(B站)](https://www.bilibili.com/video/BV17T421k78T/)
- [Sutton & Barto RL Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [OpenAI Spinning Up](https://spinningup.openai.com)

## 四阶段速览

| 阶段 | 主题 | 核心工具 |
|------|------|---------|
| 阶段一 | 建模与控制 | Pinocchio, TSID, WBC |
| 阶段二 | 仿真与训练 | IsaacGym, legged_gym, PPO |
| 阶段三 | Sim2Real | Domain Randomization, SAGE |
| 阶段四 | 进阶专题 | 按方向选：足式/loco-manip/视觉/模仿学习 |

## 推荐论文路线（按时间）

1. PPO (Schulman 2017) — RL 基础
2. AMP (Peng 2021) — 对抗模仿学习
3. ASE (Peng 2022) — 对抗技能嵌入
4. CALM (Tesslar 2023) — latent 方向控制
5. LessMimic / OmniXtreme / ULTRA (2024-2025) — 新进展

## 参考来源

- Peng et al., *AMP: Adversarial Motion Priors* (2021) — RL 模仿运动风格代表
- Peng et al., *ASE: Large-Scale Reusable Adversarial Skill Embeddings* (2022) — 技能空间嵌入
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — PPO 训练框架

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Sim2Real](../concepts/sim2real.md)
- [Locomotion](../tasks/locomotion.md)
- [开源人形机器人硬件方案对比](../entities/open-source-humanoid-hardware.md) — 低成本入门硬件选型
- **完整路线 → [[roadmap/route-a-motion-control]]**
