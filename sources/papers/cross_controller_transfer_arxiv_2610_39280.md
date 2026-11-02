# Cross-Controller Transfer（arXiv:2610.39280）

> 来源归档（paper）

- **标题：** Cross-Controller Transfer via In-Context Behavior Adaptation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39280>
- **PDF：** <https://arxiv.org/pdf/2610.39280>
- **项目页：** <https://controller-transfer.github.io/>
- **代码：** <https://github.com/JonathanYang0127/icl_cross_controller>
- **入库日期：** 2026-10-30
- **线索来源：** [机器人做错了，奖励为什么还在增加？｜附代码与项目入口](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)（具身智能小站，2026-10-30）
- **一句话说明：** 以动作指令及实际运动组成交互上下文，让统一策略适配不同控制模式与增益。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-30）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 控制器换了，策略也要读懂动作的新含义。跨控制器迁移不仅是数值换算，也可能需要调整接触任务中的操作策略。
- 以动作指令及实际运动组成交互上下文，让统一策略适配不同控制模式与增益。
- 指标：公众号转述：跨控制器迁移不仅是数值换算，也可能需要调整接触任务中的操作策略。

## 对 wiki 的映射

- [paper-cross-controller-transfer](../../wiki/entities/paper-cross-controller-transfer.md)
- [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)

## 2026-11-02 一手摘要核验（草稿）

- 依据 arXiv v1 摘要：命令与实际运动组成上下文，经 Transformer 编码；真实实验较朴素混训平均提高 33 个百分点，未见模式和增益分别为 50% 和 52%。
- **未完成：** 尚未通读全文及相关可运行源码；实验协议、局限、项目流程图与源码时序图仍需核验，不能视为完整 ingest。
