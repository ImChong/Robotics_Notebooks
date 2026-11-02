# 可变形线性物体 3D 力估计（arXiv:2610.35058）

> 来源归档（paper）

- **标题：** 3D-Force Estimation for Dual-arm Manipulation of Deformable Linear Objects via Factor Graphs
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.35058>
- **PDF：** <https://arxiv.org/pdf/2610.35058>
- **作者：** Delucchi, Annika, Di Paola, Vincenzo, Quadrelli, Bruno Marco, Zoppi, Matteo
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 基于因子图的双臂可变形线状物体三维受力估计。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 电缆、绳索等可变形线状物体没有直接驱动关节，难以通过机器人执行器信息推断物体内部或外部受到的力。
3. **周报所述创新点：** 使用因子图建立物体形状、力学约束和传感器测量之间的概率关系。在重建线状物体三维形状的同时，估计外力的大小、方向和作用位置，减少对专用应变传感器的依赖。
4. **arXiv 摘要（英文节选）：** Estimating forces acting on Deformable Linear Objects (DLOs) is challenging due to their underactuated nature. However, reliable contact estimation is crucial to precisely control them and enable safe manipulation tasks. Differently from the various approaches presented in the literature, this work adopts a probabilistic formulation where the DLO state and sensor data are related through conditional distributions.

**对 wiki 的映射**

- [paper-dlo-3d-force-estimation-factor-graphs](../../wiki/entities/paper-dlo-3d-force-estimation-factor-graphs.md)
