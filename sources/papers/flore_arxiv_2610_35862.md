# FLoRe（arXiv:2610.35862）

> 来源归档（paper）

- **标题：** Learning Force-Reactive Humanoid Loco-Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.35862>
- **PDF：** <https://arxiv.org/pdf/2610.35862>
- **项目页：** <https://louis-leee.github.io/FLoRe/>
- **代码：** <https://github.com/amazon-far/FLoRe>
- **入库日期：** 2026-10-28
- **线索来源：** [人类视频怎样变成机器人能用的动作经验？](../blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)（具身智能小站，2026-10-28）
- **一句话说明：** 用带力传感的手持夹爪与身体追踪器采集示范，策略联合预测动作和双手力命令。

## 开源状态

- **部分开放**（2026-11-05）：[官方仓库](https://github.com/amazon-far/FLoRe)已开放底层训练、权重、真机部署、数据采集软件和高层策略源码；高层训练与部署指南待发布。详见 [源码归档](../repos/flore.md)。

## 原文核查

- [arXiv v1 全文](https://arxiv.org/html/2610.35862v1)，2026-10-26：方法、实验、结论及表 I–II。
- 三个 G1 任务分别成功 7/10、9/10、8/10；同示范同控制器的位置策略合计 8/30，完整方法 24/30。仿真中 5–25 N 指令接触期误差约 1 N/手。

## 历史线索（公众号转述）

- 让人形机器人同时跟踪运动与力。混合柔顺全身控制兼顾受力与平衡，真机比较强调同示范、同控制器下的力反馈价值。
- 用带力传感的手持夹爪与身体追踪器采集示范，策略联合预测动作和双手力命令。
- 指标：公众号转述：混合柔顺全身控制兼顾受力与平衡，真机比较强调同示范、同控制器下的力反馈价值。

## 对 wiki 的映射

- [paper-flore](../../wiki/entities/paper-flore.md)
- [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md)
