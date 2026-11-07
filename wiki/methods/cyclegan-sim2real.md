---
type: method
tags: [sim2real, domain-adaptation, vision, manipulation]
status: complete
updated: 2026-11-07
related:
  - ../concepts/sim2real.md
  - ./qt-opt.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "CycleGAN 一类无配对图像翻译可把仿真渲染转成接近真实风格的观测，用于缩小视觉 Sim2Real 域差；常与大规模 RL 数据采集并列讨论。"
---

# CycleGAN 与视觉 Sim2Real

## 一句话定义

**CycleGAN**：无配对样本下的图像到图像翻译网络，通过循环一致性约束学习仿真图像与真实风格图像之间的映射；在机器人叙事中常作为**像素域对齐**工具，辅助视觉策略迁移。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 主要技术路线

- **无配对图像翻译**：学习仿真渲染图与真实风格图像之间的映射，不改动力学或控制器。
- **服务视觉闭环 RL**：与 [QT-Opt](./qt-opt.md) 等像素策略并列讨论时常用于减小观测域差；总体归类见 [Sim2Real](../concepts/sim2real.md)。
- **RL 感知的翻译（RL-CycleGAN）**：普通 CycleGAN 的翻译不感知任务，可能改掉对控制重要的细节。[RL-CycleGAN](../entities/paper-as-2011-03148-rl-cyclegan-reinforcement-learning-aware-simulat.md)（Rao et al., CVPR 2020）加入 RL-scene consistency loss，要求图像翻译前后对应的 Q 值不变，在两项视觉抓取任务上优于多种既有 sim-to-real 方法，且只需适量真实观测。

## 为什么单独成页

QT-Opt 原论文完全使用真机数据；当后续工作改用仿真训练 QT-Opt 类抓取智能体时，视觉外观差异是主要瓶颈之一；CycleGAN 代表了一条与动力学随机化并行的**表征对齐**路线。

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [QT-Opt](./qt-opt.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [RL-CycleGAN](../entities/paper-as-2011-03148-rl-cyclegan-reinforcement-learning-aware-simulat.md)
- [RCAN（Sim-to-Real via Sim-to-Sim）](../entities/paper-pai-1812-07252-simtorealviasimtosimkooseta.md)

## 参考来源

- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, https://arxiv.org/abs/1703.10593
- Rao et al., *RL-CycleGAN: Reinforcement Learning Aware Simulation-To-Real*, https://arxiv.org/abs/2006.09001
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
