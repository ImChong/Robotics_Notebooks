# HoloBoxing: A Simple Recipe for Humanoid Boxing with 7 Minutes of Single-GPU Training（arXiv:2610.32909）

> 来源归档（ingest）

- **标题：** HoloBoxing: A Simple Recipe for Humanoid Boxing with 7 Minutes of Single-GPU Training
- **类型：** paper / humanoid boxing / hierarchical control
- **arXiv：** <https://arxiv.org/abs/2610.32909>
- **项目页：** <https://hbot-competition.horizon.auto/> — [竞赛站点归档](../sites/hbot-competition.md)
- **代码：** <https://github.com/HorizonRobotics/HoloMotion/tree/master/holoboxing> — [仓库归档](../repos/horizon_robotics_holomotion.md)
- **作者：** Xihan Ma, Bo Zhang, Boxiong Li, Yuhan Bao, Ziyue Liu, Yucheng Wang
- **机构：** 地平线（Horizon Robotics）
- **提交日期：** 2026-10-23
- **入库日期：** 2026-10-27
- **一句话说明：** 在冻结的人形全身跟踪控制器之上学习紧凑的高层拳击策略，通过身体目标解码、测量状态模仿初始化和强化学习完成仿真对战；约 7 分钟仅指额外训练，不含若干前置与评估成本。
- **沉淀到 wiki：** [HoloBoxing 论文与项目实体](../../wiki/entities/holoboxing.md)

## 摘要级要点

- **分层策略：** 高层 planner 根据机器人自身状态、对手状态及比赛信息预测未来身体点目标；冻结的低层 HoloMotion 控制器将目标转成 29 个关节的执行动作。
- **紧凑目标表征：** 32 维动作经 learned decoder 映射到五个身体点（骨盆、左右腕、左右踝）的未来目标窗口。
- **初始化与优化：** 先用测量状态模仿学习初始化 planner，再通过强化学习优化交互策略；decoder 和底层控制器保持冻结。
- **时间口径：** 论文标题中的约 7 分钟是额外训练时间，含目标表征拟合复用及单 GPU 任务策略训练；不含控制器预训练、对手预训练、数据准备、评测和搜索。
- **评测范围：** 摘要报告仿真拳击实验及组件对照实验，用于评估策略行为和模仿初始化、目标表征的作用。具体指标应以论文全文为准。

## 官方资源

- 论文：<https://arxiv.org/abs/2610.32909>
- H-bot 竞赛入口：<https://hbot-competition.horizon.auto/>
- 教程与实现：<https://github.com/HorizonRobotics/HoloMotion/tree/master/holoboxing>
- 英文教程：<https://github.com/HorizonRobotics/HoloMotion/blob/master/holoboxing/README.md>
- 中文教程：<https://github.com/HorizonRobotics/HoloMotion/blob/master/holoboxing/README.zh-CN.md>

## 对 wiki 的映射

- [HoloBoxing 实体页](../../wiki/entities/holoboxing.md)
- [HoloMotion 执行控制器](../../wiki/entities/holomotion.md)
- [强化学习](../../wiki/methods/reinforcement-learning.md)
- [模仿学习](../../wiki/methods/imitation-learning.md)
