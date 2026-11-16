# BirdDreams: Joint Perspective and BEV Video Generation for a Plausible, Reactive Driving World Model

- **类型：** 论文
- **原文：** https://arxiv.org/html/2611.14815v1
- **PDF：** https://arxiv.org/pdf/2611.14815
- **版本：** arXiv:2611.14815v1（2026-11-10）
- **项目：** https://research.nvidia.com/labs/sil/projects/birddreams/
- **核对范围：** 已核对官方项目页与 arXiv HTML 的摘要、引言、方法 §3、实验章节目录；完整实验与附录尚待逐项复核。
- **通读：** 未完成
- **论文身份：** 2611.14815
- **wiki 映射：** [BirdDreams](../../wiki/entities/paper-birddreams.md)

## 原始资料要点

论文提出联合预测驾驶前视 RGB 与固定米制尺度的示意 BEV 视频。两路因果视频扩散 Transformer 通过 joint attention 交换信息，训练时两路噪声水平独立采样。可选输入自车米制轨迹：将上一帧预测 BEV 按自车位姿平移和旋转，形成下一段生成的软条件。无自车轨迹时也能生成自车运动。

训练 BEV 使用数据集地图与 3D 物体标注渲染；推理时无需 HD map、其他车辆状态或初始 BEV。官方项目页报告 PhysicalAI-AV 的轨迹跟随 ADE 5.88m → 3.83m、无动作 minADE 7.76m → 6.15m；人类比较 25 人、375 次判断，BirdDreams 57%、基线 24%、相同 19%。这些结果应按官方对照设置解读，不能视为真实道路安全保证。

## 待复核

- 全文实验设置、nuScenes 分项结果、附录和训练细节。
- 源码/权重开放情况及复现入口；截至 2026-11-16 官方项目页未展示明确的 BirdDreams 代码仓库链接。
