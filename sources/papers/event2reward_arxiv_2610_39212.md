# Event2Reward（arXiv:2610.39212）

> 来源归档（paper）

- **标题：** Event2Reward: Learning Robotic Rewards from Progress Relations
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39212>
- **PDF：** <https://arxiv.org/pdf/2610.39212>
- **项目页：** <https://niejunnan25.github.io/Event2Reward/>
- **代码：** <https://github.com/niejunnan25/Event2Reward>
- **入库日期：** 2026-10-30
- **线索来源：** [机器人做错了，奖励为什么还在增加？｜附代码与项目入口](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)（具身智能小站，2026-10-30）
- **一句话说明：** 机器人探索时会停滞甚至倒退，把示范时间当作进度容易奖励错误动作。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-30）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 让奖励分清进步、停滞与倒退。Event2Reward 用成功示范锚定进度尺度，再用失败与次优轨迹中的局部关系校正变化方向，为强化学习提供更可信的稠密奖励。
- 机器人探索时会停滞甚至倒退，把示范时间当作进度容易奖励错误动作。
- 指标：公众号转述：三项精密插入真机任务、30 分钟训练后平均成功率 60.0%，对比 Robometer 为 37.8%（每任务每条件三次训练，每个检查点评估二十回合）；新增的是对停滞、倒退和恢复事件的局部进度关系监督，无需为次优轨迹逐帧指定绝对进度。

## 对 wiki 的映射

- [paper-event2reward](../../wiki/entities/paper-event2reward.md)
- [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
