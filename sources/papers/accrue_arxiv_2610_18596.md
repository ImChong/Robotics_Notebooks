# ACCRUE: Continual Off-Policy Reinforcement Learning for Robot Foundation Models（arXiv:2610.18596）

> 来源归档（ingest）

- **短名：** ACCRUE
- **类型：** paper
- **作者：** Changyeon Kim、Kimin Lee、Yashraj Narang、Yijie Guo
- **arXiv：** <https://arxiv.org/abs/2610.18596>（v1，2026-10-13）
- **项目页：** <https://accrue-rl.github.io/>
- **代码：** 截至 2026-10-14，项目页未列公开代码仓库
- **入库日期：** 2026-10-14
- **一句话说明：** 在机器人基础模型上跨任务继承轻量 actor/critic 与小型保留回放集，以离策略方式提高后续任务学习效率并缓解遗忘。

## 核心摘录（面向 wiki 编译）

1. **目标：** 连接持续学习的跨任务保留能力与离策略 RL 的样本效率，避免每个新任务都从头开始或只做 on-policy 更新。
2. **继承机制：** 轻量 actor 与 critic 跨任务阶段传递；继承 critic 只用当前任务转移适配，并指导当前 actor 学习。
3. **抗遗忘：** 缓存少量旧任务 episode 的 actor 目标，通过蒸馏保持旧行为；固定视觉特征让旧回放与继承学习器保持兼容。
4. **评测：** 在仿真和真机接触丰富装配序列上，后续任务学习更快、遗忘接近零；真机训练时间减少 31.0%–68.5%。
5. **边界：** 项目页未列公开代码入口；摘要未提供任务数、基线细节或每项硬件实验分母，不能仅凭总幅度推断跨平台泛化。

**对 wiki 的映射：** [paper-accrue-continual-off-policy-rl](../../wiki/entities/paper-accrue-continual-off-policy-rl.md)

## 版本与引用

- arXiv v1，提交于 2026-10-13。
- 来源以 arXiv 摘要和 ACCRUE 项目页为准；arXiv 元数据未列作者机构。
---
