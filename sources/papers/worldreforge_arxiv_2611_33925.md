# WorldReforge：从仿真失败学习机器人世界模型

- **论文：** [arXiv:2611.33925v1](https://arxiv.org/abs/2611.33925v1) · [原文 HTML](https://arxiv.org/html/2611.33925v1) · [PDF](https://arxiv.org/pdf/2611.33925v1)
- **原题：** Beyond the Happy Path: Generalizing Robot World Models from Simulated Failures
- **作者：** Zhiyue Zhao、Mingyu Liu、Bolin Zhang、Liyang Li、Guangyu Huang、Canyu Zhao、Jinchuan Shi、Haokai Zhang、Hao Chen、Chunhua Shen
- **机构：** 浙江大学、上海创智学院
- **发布：** 2026-11-25（arXiv v1）
- **入库日期：** 2026-11-27
- **核对范围：** 通读 v1 正文第 1–6 节与附录 A–E；核对训练数据和评测切分、两个训练损失、接触失败与保持的互补性、表 1 与成对动作误差、几何投影及 ADE/FDE 计量方式。
- **通读：** 已完成
- **项目：** [项目页](../sites/worldreforge.md)；[官方代码](../repos/worldreforge.md)
- **Wiki：** [WorldReforge](../../wiki/entities/paper-worldreforge.md)

作者用参考轨迹加有控制扰动造出多种抓取后果，记录 RGB 与仿真物理真值。公开 Reforge-Eval 对应仿真评测切分，不将其与完整训练语料或真实失败切分混用。Reforge 指仅改数据的基础目标后训练，WorldReforge 则在相同训练预算下增加 IA 和 SC；论文真实视频结果是重建指标，不是闭环控制结果。
