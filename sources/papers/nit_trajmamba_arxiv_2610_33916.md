# NIT-TrajMamba（arXiv:2610.33916）

> 来源归档（paper）

- **标题：** Real-Time Human Trajectory Prediction from RGB-D for Mobile Robots Using Bi-Mamba
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.33916>
- **PDF：** <https://arxiv.org/pdf/2610.33916>
- **代码：** <https://github.com/basheeraltawil/NIT-TrajMamba>
- **入库日期：** 2026-10-28
- **线索来源：** [人类视频怎样变成机器人能用的动作经验？](../blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)（具身智能小站，2026-10-28）
- **一句话说明：** 统一训练与部署中的姿态检测、深度反投影和平滑流程，再用双向 Mamba 与解码器预测短期速度。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-28）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 让行人轨迹预测贴近真实输入。部署于移动机器人，重点是减少理想训练数据与噪声传感输入的差距。
- 统一训练与部署中的姿态检测、深度反投影和平滑流程，再用双向 Mamba 与解码器预测短期速度。
- 指标：公众号转述：部署于移动机器人，重点是减少理想训练数据与噪声传感输入的差距。

## 对 wiki 的映射

- [paper-nit-trajmamba](../../wiki/entities/paper-nit-trajmamba.md)
- [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md)
