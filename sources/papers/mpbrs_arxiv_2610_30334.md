# MPBRS：人形移动的多势函数奖励塑形（arXiv:2610.30334）

> 来源归档（ingest）

- **标题：** MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion
- **简称：** MPBRS
- **类型：** paper / humanoid / locomotion / reinforcement-learning / reward-shaping / sim2sim
- **arXiv：** <https://arxiv.org/abs/2610.30334>（v1，2026-10-21）
- **作者：** Van Chung Nguyen、Hung Manh La
- **项目页 / 代码：** arXiv 页面未列出官方项目页或代码链接；开放状态待核实
- **入库日期：** 2026-10-24
- **一句话说明：** 评估多势函数奖励塑形在九种人形机器人上的统一配置，强调跨机器人复用相同奖励结构与权重，仅需机器人特定的关节/连杆映射和默认姿态元数据。

## 摘要级要点

- **动机：** 手工奖励项可能互相冲突、过拟合单一机器人，并对权重微调敏感。
- **方法：** 研究 Multi-Potential-Based Reward Shaping（Multi-PBRS），通过多个势函数的差分塑形奖励，提出统一的人形移动奖励配置。
- **跨机器人评估：** 在九种人形机器人上沿用相同奖励结构与权重；只针对每台机器人提供关节/连杆映射和默认姿态等元数据，不为每台机器人重新设计奖励。
- **摘要结论：** 相比标准 shaping，MPBRS 表现出更一致的缩放行为，并降低对调参的敏感性；具体机器人、指标与结果范围以论文全文为准。

## 论文入口

- [arXiv 摘要](https://arxiv.org/abs/2610.30334)
- [arXiv PDF](https://arxiv.org/pdf/2610.30334)

## 对 wiki 的映射

- 实体页：[paper-mpbrs-humanoid-reward-shaping.md](../../wiki/entities/paper-mpbrs-humanoid-reward-shaping.md)
- 相关主题：奖励设计、PBRS、跨具身强化学习、人形 locomotion
