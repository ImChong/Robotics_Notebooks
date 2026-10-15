# UMV 官方说明页（RAI Institute）

> 来源归档（ingest 配套站点）

- **标题：** Designing Wheeled Robotic Systems for Ultra Mobility and Performance
- **类型：** site / 官方项目说明
- **URL：** <https://rai-inst.com/resources/blog/designing-wheeled-robotic-systems/>
- **机构：** 机器人与人工智能研究所（RAI Institute）
- **论文：** [Science Robotics DOI](https://doi.org/10.1126/scirobotics.aed3627)；[arXiv:2602.22118](https://arxiv.org/abs/2602.22118)
- **代码 / 权重：** 页面未提供训练、推理或部署代码 / 权重链接（核查日期：2026-10-15）
- **入库日期：** 2026-10-15
- **一句话说明：** RAI 官方文章介绍 UMV 的高位反作用质量、传感器、Isaac Lab RL 训练及 sim-to-real 流程；强调论文中的实验室动捕辅助与户外机载估计演示需区分。

## 页面内容要点

- **机构设计：** 通过高位 Head 与可动空间连杆调整整机质量分布；跳跃时扩展至约 1.52 m、空中收拢车体。
- **传感器：** 关节编码器、IMU、高速 ToF、LiDAR，以及实验室定位相机/视觉标记。
- **学习控制：** Isaac Lab 仿真试错训练策略，实机观察后回到仿真调整奖励；通过模型校准和物理随机化缩小 sim-to-real 差异。
- **开放核查：** 页面含技术介绍与演示，没有链接到 UMV 训练/部署代码或权重；不把 RAI 其他开源项目误认为 UMV 源码。

## 对 wiki 的映射

- [UMV 论文实体](../../wiki/entities/paper-umv-ultra-mobility-vehicle.md)
- [论文摘录](../papers/umv_arxiv_2602_22118.md)
