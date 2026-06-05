---
type: method
tags: [rl, manipulation, vision, continuous-actions, off-policy, google-robotics]
status: complete
updated: 2026-05-10
related:
  - ./reinforcement-learning.md
  - ./mt-opt.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "QT-Opt 用大规模离线 + 在线真实机器人数据训练视觉条件下的连续动作 Q 学习，是早期证明「像素操控可扩展」的代表系统之一。"
---

# QT-Opt

## 一句话定义

**QT-Opt**：面向视觉输入机械臂抓取的异策深度强化学习框架，用交叉熵方法等近似在连续动作空间上做 Q 学习，并结合长时间运行的真实机器人数据采集闭环。

## 主要技术路线

- **连续动作 Q 学习**：区别于离散动作 Atari，在连续关节 / 末端空间估计 Q；整体定位见 [Sim2Real](../concepts/sim2real.md) 中的像素闭环叙事。
- **规模化真实数据**：依赖并行机器人采集与再训练循环；配套博客描述了分布式训练与评估栈。
- **仿真—真实视觉对齐**：常配合图像域迁移（例如 [CycleGAN Sim2Real](./cyclegan-sim2real.md)）缩小观测差异。

## 关联页面

- [MT-Opt](./mt-opt.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Manipulation 任务](../tasks/manipulation.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 参考来源

- Kalashnikov et al., *Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation*, https://arxiv.org/abs/1806.10293
- Google Research Blog: https://blog.research.google/2018/06/scalable-deep-reinforcement-learning.html
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
