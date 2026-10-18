# RoboMP-DINOv2（arXiv:2609.25506）

> 来源归档（paper）

- **标题：** RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25506>
- **PDF：** <https://arxiv.org/pdf/2609.25506>
- **代码：** <https://github.com/han20192019/RoboMP_DINOv2>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 在完整图像特征上加入区域提示，而不是用硬掩码过滤，避免丢掉障碍与背景线索。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 硬掩码会丢掉障碍和背景线索；区域提示保留完整场景。
- 完整图像特征 + 区域提示（空间提示），替代硬掩码。
- 指标：公众号称空间、杂乱和颜色迁移实验均在仿真中完成，颜色增强版本应单独看待。

## 对 wiki 的映射

- [paper-robomp-dinov2](../../wiki/entities/paper-robomp-dinov2.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
