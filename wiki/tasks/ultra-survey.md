---
type: task
sources:
  - ../../sources/papers/survey_papers.md
  - ../../sources/papers/locomotion_rl.md
---

# ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation

**统一多模态控制：实现人形机器人自主全身移动操作**

## 一句话定义

ULTRA 是一个**统一的多模态控制器**——有动作参考时能精确跟踪，没有参考时也能从第一人称视觉感知和简单的任务指令自主生成全身移动操作行为。

## 核心贡献

| 创新 | 描述 |
|------|------|
| **Neural Retargeting** | 一个 RL 策略搞定所有人类动捕→机器人动作的转换，保证物理可行性 |
| **双模式控制** | 同一个控制器支持"跟踪参考"和"自主行为"两种模式 |
| **Latent Skill Space** | 把大量运动技能压缩到低维隐空间（64维），方便高层策略调用 |
| **Egocentric Perception** | 从第一人称深度图感知环境，不依赖外部传感器 |

## 解决的问题

1. **数据稀缺**：人类动捕数据丰富，但人与机器人身体结构不同，简单 retargeting 产生物理不可行动作
2. **技能扩展难**：以前一个任务训一个策略，ULTRA 用统一策略处理所有动作
3. **依赖预定义参考**：ULTRA 支持从感知和高层指令自主生成行为，不需预录参考动作

## 关键技术

### Retargeting = RL 问题
- 训练一个神经网络策略，输入人类动捕动作，输出机器人关节指令
- 在物理仿真器（Isaac Gym）中训练，奖励包含：跟踪误差 + 物理稳定性 + 接触一致性
- 核心：一个策略处理所有动作，不需要每个动捕片段单独训练

### 统一控制器三阶段
1. **通用跟踪策略蒸馏**：Teacher-Student 蒸馏成统一模型
2. **运动技能压缩到隐空间**：VAE 把技能编码进 64 维隐空间
3. **RL 微调**：增强 OOD 场景鲁棒性

### Sim-to-Real
- 域随机化（物理参数、观测噪声）
- 课程学习：从精确状态逐步过渡到第一人称深度感知

## 工程细节

- **训练仿真器**：Isaac Gym（GPU 并行）
- **验证仿真器**：MuJoCo
- **真机平台**：Unitree G1
- **数据集**：OMOMO（人-物交互动捕，非 AMASS）
- **算法**：PPO，Actor-Critic 分离，3层 MLP [1024, 1024, 512]
- **Student 网络**：Transformer（2层，4头）+ 64维隐变量 z + FiLM 调制

## 相关页面

- [Locomotion](./locomotion.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Unitree](../entities/unitree.md)
- [Isaac Gym / Isaac Lab](../entities/isaac-gym-isaac-lab.md)
- [Sim2Real](../concepts/sim2real.md)

## 论文信息

| 项目 | 链接 |
|------|------|
| **arXiv** | [2603.03279](https://arxiv.org/abs/2603.03279) |
| **项目主页** | [ultra-humanoid.github.io](https://ultra-humanoid.github.io/) |
| **发布时间** | 2026年3月3日 |
| **机构** | UIUC |

## 参考来源

- arXiv: [2603.03279](https://arxiv.org/abs/2603.03279) — ULTRA 原论文
- 项目主页：[ultra-humanoid.github.io](https://ultra-humanoid.github.io/)

## 关联页面

- [Locomotion](./locomotion.md) — ULTRA 的运动控制子任务
- [Loco-Manipulation](./loco-manipulation.md) — ULTRA 解决的核心任务类型
- [Imitation Learning](../methods/imitation-learning.md) — ULTRA 基于 IL 框架

## 推荐继续阅读

详细笔记（含面试高频问题、工程复现要点、DAgger 通俗解释）见：[Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)
