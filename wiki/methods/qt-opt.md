---
type: method
tags: [rl, manipulation, vision, continuous-actions, off-policy, google-robotics]
status: complete
updated: 2026-11-07
related:
  - ./reinforcement-learning.md
  - ./mt-opt.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "QT-Opt 用 7 台真机采集的 58 万余次抓取（离线数据 + 少量在线联合微调）训练视觉条件下的连续动作 Q 学习，未见物体抓取成功率 96%，是早期证明「像素操控可扩展」的代表系统之一。"
---

# QT-Opt

## 一句话定义

**QT-Opt**：面向视觉输入机械臂抓取的异策深度强化学习框架，不训练显式 actor，而是用交叉熵方法（CEM）在 Q 函数上做随机优化来选动作与算目标值，从而在连续动作空间上做 Q 学习，并结合长时间运行的真实机器人数据采集闭环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 主要技术路线

- **连续动作 Q 学习**：区别于离散动作 Atari，动作为末端执行器笛卡尔位移、绕竖直轴转角与夹爪开合等命令，用 CEM 对 Q 取近似最大；整体定位见 [Sim2Real](../concepts/sim2real.md) 中的像素闭环叙事。
- **规模化真实数据**：论文在 7 台 KUKA 机械臂上数周内采集 58 万余次抓取，仅用肩上视角 RGB 图像，奖励为「成功提起物体」的二值信号；完全离线训练已优于基线，再加少量在线联合微调后未见物体抓取成功率达 96%，并出现重抓、试探、先挪物体再抓等涌现行为。配套博客描述了分布式训练与评估栈。
- **仿真—真实视觉对齐（后续工作）**：QT-Opt 原论文完全在真机数据上训练（仿真只用于消融）；后续工作把 QT-Opt 当作仿真训练的抓取智能体，再用图像域迁移缩小观测差异，例如 [RCAN](../entities/paper-pai-1812-07252-simtorealviasimtosimkooseta.md) 与 [RL-CycleGAN](../entities/paper-as-2011-03148-rl-cyclegan-reinforcement-learning-aware-simulat.md)（见 [CycleGAN Sim2Real](./cyclegan-sim2real.md)）。

## 关联页面

- [MT-Opt](./mt-opt.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Manipulation 任务](../tasks/manipulation.md)
- [CycleGAN Sim2Real](./cyclegan-sim2real.md)
- [RL-CycleGAN](../entities/paper-as-2011-03148-rl-cyclegan-reinforcement-learning-aware-simulat.md)
- [RCAN（Sim-to-Real via Sim-to-Sim）](../entities/paper-pai-1812-07252-simtorealviasimtosimkooseta.md)

## 参考来源

- Kalashnikov et al., *Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation*, https://arxiv.org/abs/1806.10293
- Google Research Blog: https://blog.research.google/2018/06/scalable-deep-reinforcement-learning.html
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
