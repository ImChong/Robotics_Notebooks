---
type: reference_page
summary: "按主题整理机器人学（locomotion, manipulation, learning control）主线核心论文入口。"
updated: 2026-04-23
---

# 论文导航 / Papers

这里不是论文全文仓库，也不是逐篇精读笔记区。

`papers/` 的职责是：

> 按主题整理当前主线最值得继续看的论文入口，让你知道“这个方向往下该看哪些论文”。

如果你想做单篇论文深读，请去：
- [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)

---

## 适合谁看

适合：
- 你已经在 `wiki/` 里搞懂一个概念的大致意思
- 现在想顺着这个主题继续看论文
- 不知道该从哪里开始补 reading list

不适合：
- 你还不知道这个概念本身是什么
- 你想直接读论文原文总结（那是论文笔记项目的职责）

---

## 快速入口

| 你的目标 | 从这里进入 |
|---------|-----------|
| 想看 locomotion RL 论文 | [Locomotion RL](locomotion-rl.md) |
| 想看模仿学习论文 | [Imitation Learning](imitation-learning.md) |
| 想看 WBC / TSID / whole-body 论文 | [Whole-Body Control](whole-body-control.md) |
| 想看 sim2real 论文 | [Sim2Real](sim2real.md) |
| 想看人形硬件论文与平台脉络 | [Humanoid Hardware](humanoid-hardware.md) |
| 想先看综述再扩展 | [Survey Papers](survey-papers.md) |

---

## 当前主线怎么对应到 papers/

### 控制 / 优化主线
如果你在看：
- Centroidal Dynamics
- Trajectory Optimization
- MPC
- TSID / WBC

建议从这里进入：
- [Whole-Body Control](whole-body-control.md)
- [Survey Papers](survey-papers.md)

### 学习 / locomotion 主线
如果你在看：
- Reinforcement Learning
- Locomotion
- Imitation Learning

建议从这里进入：
- [Locomotion RL](locomotion-rl.md)
- [Imitation Learning](imitation-learning.md)

### sim2real / 平台主线
如果你在看：
- Sim2Real
- System Identification
- Domain Randomization

建议从这里进入：
- [Sim2Real](sim2real.md)
- [Survey Papers](survey-papers.md)

### 硬件 / 人形平台主线
如果你在看：
- Unitree
- humanoid platforms
- research hardware ecosystems

建议从这里进入：
- [Humanoid Hardware](humanoid-hardware.md)

---

## 当前判断

`papers/` 这一层后续的重点不是无限加论文列表，而是：

1. 让每个主题页更能回答“先看哪几篇”
2. 和 wiki 主线对应得更显式
3. 控制“论文入口”和“单篇深读”之间的边界
