# TACOS（arXiv:2610.37806）

> 来源归档（paper）

- **标题：** TACOS: Terrain-Aware COntact Supervision for Visual Footstep Planning
- **作者：** Jean-Michel Fortin、William Larrivée-Hardy、Gabriel Jeanson、Olivier Gamache、Daniele De Martini、François Pomerleau、Philippe Giguère
- **机构：** 拉瓦尔大学 Northern Robotics Laboratory；牛津大学 Mobile Robotics Group
- **版本：** [arXiv v1，2026-10-27](https://arxiv.org/html/2610.37806v1)
- **PDF：** <https://arxiv.org/pdf/2610.37806v1>
- **代码：** <https://github.com/norlab-ulaval/TACOS>
- **入库日期：** 2026-10-30；原文核对：2026-11-04
- **核对范围：** 论文正文的方法、实验、结论与局限。

## 主要内容

- 因子图接触残差构成每步连续不稳定评分，Hard 使用后验优化，Soft 的残差可因果计算。
- 将足端评分投影到落脚前图像，以稀疏监督训练 DINOv3-DPT 预测器；可加入相机姿态及近期不稳定历史。
- 1716 次人工标注接触中，Hard 和 Soft AP 分别为 0.96、0.93；50 米森林闭环实验为有限示范。
- 系统依赖准确的接触检测、状态估计和时间同步；作者将深度、几何及持续学习列为后续方向。

## 对 wiki 的映射

- [TACOS 实体](../../wiki/entities/paper-tacos.md)
- [同期论文阅读地图](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
