---
type: entity
project_id: accrue-continual-off-policy-rl
tags: [paper, reinforcement-learning, continual-learning, off-policy, manipulation]
status: complete
updated: 2026-10-14
arxiv: "2610.18596"
project: https://accrue-rl.github.io/
summary: "ACCRUE 在任务间继承轻量 actor/critic，以当前任务离策略更新结合少量旧 episode 蒸馏；真机接触装配训练时间报告减少 31.0%–68.5%，近零遗忘。"
related:
  - ../methods/reinforcement-learning.md
  - ../methods/model-based-rl.md
  - ../concepts/contact-rich-manipulation.md
  - ./paper-streaming-rl-continual-robotics.md
  - ./paper-experience-driven-continual-learning-of.md
sources:
  - ../../sources/papers/accrue_arxiv_2610_18596.md
  - ../../sources/sites/accrue-project.md
---

# ACCRUE：机器人基础模型的持续离策略强化学习

**ACCRUE**（*Continual Off-Policy Reinforcement Learning for Robot Foundation Models*，arXiv:2610.18596）让轻量 actor 与 critic 跨连续任务继承，并用少量旧任务 episode 保持既有行为，从而提高后续机器人任务的离策略学习效率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACCRUE | Continual Off-Policy Reinforcement Learning for Robot Foundation Models | 本文持续离策略后训练框架 |
| RL | Reinforcement Learning | 用任务奖励更新机器人策略的方法 |
| RFM | Robot Foundation Model | 可适配多个机器人任务的基础策略模型 |
| VLA | Vision-Language-Action | 论文讨论的机器人基础模型策略类别 |
| OPRL | Off-Policy Reinforcement Learning | 可复用历史转移数据进行更新的强化学习方式 |

## 为什么重要

机器人策略上线后会逐步遇到新任务。逐个任务从头训练浪费已有经验；持续学习若只用 on-policy 更新，又可能采样昂贵且跨任务迁移弱。ACCRUE 将轻量 actor/critic、离策略数据复用和旧行为保持组合起来，针对“学新任务时如何少忘旧技能”提供一条训练路线。

## 方法

1. **跨任务继承学习器。** 每个任务阶段把轻量 actor 和 critic 传给下一任务，而不是重置全部 RL 后训练状态。
2. **critic 适配当前任务。** 继承的 critic 只使用当前任务转移更新，再指导当前任务 actor 学习，减少新旧任务价值目标直接混用。
3. **保留少量旧任务行为。** 缓存少数旧 episode 的 actor 输出作为目标，通过蒸馏约束新 actor，防止遗忘。
4. **固定视觉特征。** 视觉表示保持稳定，使继承学习器与回放数据兼容，减少因表征漂移造成的旧数据失效。

## 实验与评测

论文在仿真与真机的接触丰富装配任务序列上评估。摘要报告 ACCRUE 加快后续任务学习、遗忘接近零，并使真机训练时间减少 **31.0%–68.5%**。公开摘要未列任务数、基线名称、样本量和分硬件结果；这些细节应以论文全文实验表为准。

## 结论

**ACCRUE 把跨任务持续性压缩为可继承的轻量学习器和小规模行为记忆，适用于昂贵真机任务按序扩展的设定。**

1. **继承 actor 和 critic，而非只继承初始策略。** critic 的当前任务适配为新任务 actor 提供可复用学习信号。
2. **用小回放集保持旧行为。** 蒸馏旧 episode 目标可缓解遗忘，存储预算仍需根据任务覆盖验证。
3. **视觉特征稳定是系统假设。** 固定视觉特征支持回放兼容；若更换编码器或观测分布变化，应重新评估旧数据价值。
4. **真机时间收益值得关注，但区间不能独立解释。** 31.0%–68.5% 的幅度需结合具体任务、基线和训练时间定义。
5. **源码暂未公开。** 项目页没有仓库入口，目前无法复核运行配置或复现实验。

## 与其他工作对比

| 对照 | 差异 |
|------|------|
| [Streaming RL for Continual Robotics](./paper-streaming-rl-continual-robotics.md) | 两者都关注连续任务与遗忘；ACCRUE 主张在机器人基础模型上跨任务继承 actor/critic 并用离策略转移学习 |
| [Experience-Driven Continual Learning](./paper-experience-driven-continual-learning-of.md) | ACCRUE 将持续性用于机器人基础模型的 RL 后训练，并保留小型旧行为回放 |
| [Model-Based RL](../methods/model-based-rl.md) | ACCRUE 是策略与价值网络跨任务复用的 model-free 路线，不依赖显式世界模型规划 |

## 工程实践

- **按阶段记录：** 每个任务阶段保存 actor/critic 版本、当前任务转移、保留 episode 及蒸馏损失，便于定位新任务收益和旧任务退化。
- **先量化遗忘：** 在每轮新任务训练后重跑全部旧任务，报告前向迁移、后向迁移和训练样本数。
- **约束回放内存：** 逐步扫描保留 episode 数量，量化旧行为保持和当前任务样本效率之间的折中。
- **固定视觉表征需验证：** 更换视觉 encoder、相机布局或光照范围后，应测试表征冻结是否仍保持回放兼容。
- **源码状态：** 项目页未列官方代码链接，当前不适用源码运行时序图。

## 局限与风险

- 结果来自接触丰富装配任务序列，不能直接推出对开放任务顺序、不同机器人本体均有效。
- 固定视觉特征有助于回放兼容，但可能限制遇到新视觉域时的表征适应能力。
- 摘要没有给出完整基线和逐任务统计，训练时间节省幅度应谨慎解读。

## 关联页面

- [强化学习](../methods/reinforcement-learning.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Streaming RL for Continual Robotics](./paper-streaming-rl-continual-robotics.md)

## 参考来源

- [ACCRUE 论文摘录](../../sources/papers/accrue_arxiv_2610_18596.md)
- [ACCRUE 项目页核查](../../sources/sites/accrue-project.md)
- [arXiv:2610.18596](https://arxiv.org/abs/2610.18596)

## 推荐继续阅读

- [ACCRUE 项目页](https://accrue-rl.github.io/)
- [arXiv:2610.18596](https://arxiv.org/abs/2610.18596)
