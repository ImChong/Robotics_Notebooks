---
type: method
tags: [rl, multi-task, manipulation, google-robotics, off-policy]
status: complete
updated: 2026-11-07
related:
  - ./qt-opt.md
  - ./reinforcement-learning.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "MT-Opt 在 QT-Opt 基础上把连续动作机器人强化学习扩展到多任务：7 台真机共享采集 12 项操控任务数据，用视觉成功检测器给奖励并通过跨任务经验共享提升样本效率。"
---

# MT-Opt

## 一句话定义

**MT-Opt**：在多机器人并行采集框架下，同时学习多项操控技能的连续动作多任务深度强化学习系统；强调任务规范、成功检测器与跨任务表示共享。算法上扩展自单任务的 [QT-Opt](./qt-opt.md)；新任务可通过用户提供的期望结果示例来定义。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| AI | Artificial Intelligence | 人工智能 |

## 主要技术路线

- **多任务并行 RL**：在共享机器人农场与任务判定器上同时优化多项操控技能，共享表示与跨任务经验。论文在 7 台真机上学习 12 项任务（语义抓取、放入固定位、覆盖、对齐、重排等），由多任务视觉成功检测器（SD）判定稀疏奖励。
- **任务冒充（task impersonation）与数据再平衡**：把为某任务采集的回合按成功检测器重新标注给其他相关任务使用，并控制各任务的成功 / 失败样本比例；论文报告可泛化到结构相似的新任务，并借助既有经验更快学会新任务。
- **与 QT-Opt 对照**：[QT-Opt](./qt-opt.md) 侧重单技能规模化抓取；MT-Opt 强调任务并行与迁移；二者同属连续视觉 RL，可与 [Sim2Real](../concepts/sim2real.md) 中的真实数据闭环对照阅读。

## 关联页面

- [QT-Opt](./qt-opt.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [SayCan](../entities/paper-saycan.md)
- [具身三个时代叙事](../queries/robot-learning-three-eras-narrative.md)

## 参考来源

- Kalashnikov et al., *MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale*, https://arxiv.org/abs/2104.08212
- Google AI Blog: https://ai.googleblog.com/2021/04/multi-task-robotic-reinforcement.html
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
