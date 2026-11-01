# QbD-VLA（arXiv:2610.38661）

> 来源归档（paper）

- **标题：** Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.38661>
- **PDF：** <https://arxiv.org/pdf/2610.38661>
- **代码：** <https://github.com/jiuyixu25/QbD-VLA>
- **入库日期：** 2026-10-30
- **线索来源：** [机器人做错了，奖励为什么还在增加？｜附代码与项目入口](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)（具身智能小站，2026-10-30）
- **一句话说明：** QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-30）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 按任务表现分配 VLA 的量化精度。仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。
- QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。
- 指标：公众号转述：仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。

## 对 wiki 的映射

- [paper-qbd-vla](../../wiki/entities/paper-qbd-vla.md)
- [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
