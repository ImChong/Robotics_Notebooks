# Dont Risk It（arXiv:2610.40050）

> 来源归档（paper）

- **标题：** Don’t Risk It!: Trajectory-Conditioned Distributional Risk for LiDAR-Only Crowd Navigation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40050>
- **PDF：** <https://arxiv.org/pdf/2610.40050>
- **项目页：** <https://dont-risk-it-planner.github.io/>
- **代码：** <https://github.com/dont-risk-it-planner/drip>
- **入库日期：** 2026-11-03
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** DRIP 直接使用激光雷达占据历史，预测候选动作序列对应的最小间距分布，并以碰撞结果约束风险学习，减少对易失效目标跟踪的依赖。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-03）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 在人群中按轨迹预测碰撞风险。
- DRIP 直接使用激光雷达占据历史，预测候选动作序列对应的最小间距分布，并以碰撞结果约束风险学习，减少对易失效目标跟踪的依赖。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-dont-risk-it-crowd-navigation](../../wiki/entities/paper-dont-risk-it-crowd-navigation.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
