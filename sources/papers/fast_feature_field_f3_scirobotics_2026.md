# Fast Feature Field (F³): A general predictive representation of events for robot perception

> 来源归档（ingest）

- **作者：** Richeek Das、Kostas Daniilidis、Pratik Chaudhari
- **机构：** 宾夕法尼亚大学 GRASP Laboratory
- **期刊：** Science Robotics, Vol. 11, Issue 119, 2026-10-14
- **DOI：** <https://doi.org/10.1126/scirobotics.aec8218>
- **预印本：** <https://arxiv.org/abs/2509.25146>（预印本标题为 *Fast Feature Field (F³): A Predictive Representation of Events*）
- **项目页：** <https://www.seas.upenn.edu/~richeek/f3/>
- **代码：** <https://github.com/grasp-lyrl/fast-feature-fields>
- **入库日期：** 2026-10-15
- **一句话说明：** 从过去事件学习能预测未来事件的特征表示，并将其用于光流、语义分割、立体匹配和单目度量深度等机器人视觉任务。

## 摘要级要点

- F³ 把连续时空窗口内的事件编码为多通道特征图；作者将其描述为过去事件的统计量，用于预测未来事件，同时保留场景结构与运动信息。
- 多分辨率哈希编码利用事件数据稀疏性，并结合排列不变的 Deep Sets 架构处理事件集合。
- 项目页报告表示计算速度为 HD 120 Hz、VGA 440 Hz；下游任务在 HD 下约 25–75 Hz。
- 项目页展示汽车、四足机器人与飞行平台数据，覆盖昼夜、室内外、城市与非铺装环境及不同传感器分辨率和事件率。
- 官方仓库提供预训练推理 notebook、PyTorch Hub 加载、训练入口、下游任务实现及 AOTI 导出；数据集需依文档另行配置。

## 对 wiki 的映射

- 主实体：[F³ 论文实体页](../../wiki/entities/paper-fast-feature-field-f3.md)
- 项目页：[F³ 项目页归档](../sites/fast-feature-field-f3.md)
- 代码：[官方仓库导航](../repos/fast-feature-fields.md)
