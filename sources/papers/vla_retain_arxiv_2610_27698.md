# VLA-Retain（arXiv:2610.27698）

> 来源归档（paper）

- **标题：** VLA-Retain: Benchmarking Cross-Dimension Effects of Vision Tuning in VLAs
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.27698>
- **PDF：** <https://arxiv.org/pdf/2610.27698>
- **代码：** <https://github.com/BoruiXu/VLA_vision_tuning_benchmark>
- **入库日期：** 2026-10-31
- **线索来源：** [一张照片变成仿真场景，机器人真能用吗？｜附代码与项目入口](../blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)（具身智能小站，2026-10-31）
- **一句话说明：** 换相机或降噪后的成功率变高，仍可能牺牲其他场景。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-31）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 适配一个视角，会不会损失其他能力。VLA-Retain用同一模型调优前后的配对回合，把目标收益和非目标退化分开，适合部署验收与适配方案选择。
- 换相机或降噪后的成功率变高，仍可能牺牲其他场景。
- 指标：公众号转述：五种VLA在七类扰动、每配置10,030回合上比较；十个模型—适配域组合中，8个出现非目标维度平均收益为负。

## 对 wiki 的映射

- [paper-vla-retain](../../wiki/entities/paper-vla-retain.md)
- [单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-31-50-papers-technology-map.md)
