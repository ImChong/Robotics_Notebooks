---
type: method
tags: [rl, multi-task, manipulation, google-robotics, off-policy]
status: complete
updated: 2026-05-10
related:
  - ./qt-opt.md
  - ./reinforcement-learning.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "MT-Opt 把多任务机器人强化学习扩展到连续动作与共享数据采集栈，用任务判定与跨任务经验共享提升样本效率。"
---

# MT-Opt

## 一句话定义

**MT-Opt**：在多机器人并行采集框架下，同时学习多项操控技能的连续动作多任务深度强化学习系统；强调任务规范、成功检测器与跨任务表示共享。

## 主要技术路线

- **多任务并行 RL**：在共享机器人农场与任务判定器上同时优化多项操控技能，共享表示与跨任务经验。
- **与 QT-Opt 对照**：[QT-Opt](./qt-opt.md) 侧重单技能规模化抓取；MT-Opt 强调任务并行与迁移；二者同属连续视觉 RL，可与 [Sim2Real](../concepts/sim2real.md) 中的真实数据闭环对照阅读。

## 关联页面

- [QT-Opt](./qt-opt.md)
- [Reinforcement Learning](./reinforcement-learning.md)

## 参考来源

- Gupta et al., *MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale*, https://arxiv.org/abs/2104.08212
- Google AI Blog: https://ai.googleblog.com/2021/04/multi-task-robotic-reinforcement.html
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
