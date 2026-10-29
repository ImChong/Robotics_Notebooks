# Rendering-Free Event Sim（arXiv:2610.37543）

> 来源归档（paper）

- **标题：** Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous Robot Navigation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37543>
- **PDF：** <https://arxiv.org/pdf/2610.37543>
- **代码：** <https://github.com/tudelft/rendering-free-event-nav>
- **入库日期：** 2026-10-28
- **线索来源：** [人类视频怎样变成机器人能用的动作经验？](../blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)（具身智能小站，2026-10-28）
- **一句话说明：** 事件相机适合低功耗、低延迟机器人，但先渲染图像再转换事件的仿真流程会拖慢强化学习。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-28）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 让微型无人机绕开事件仿真瓶颈。本文从运动边缘直接生成事件观测，把训练吞吐与微控制器部署连起来，值得关注的是整条工程路径，以及近似传感模型能否支持真实迁移。
- 事件相机适合低功耗、低延迟机器人，但先渲染图像再转换事件的仿真流程会拖慢强化学习。
- 指标：公众号转述：把闭环事件观测本身做成可并行解析近似，而非仅优化渲染或用特权状态替代事件输入；再将传感近似、策略训练与机载部署放在同一验证链条中。

## 对 wiki 的映射

- [paper-rendering-free-event-sim](../../wiki/entities/paper-rendering-free-event-sim.md)
- [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md)
