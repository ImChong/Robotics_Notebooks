# An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems.html>
- **分类：** 09_State_Estimation
- **arXiv：** <https://arxiv.org/abs/2207.06780>
- **入库日期：** 2026-06-07
- **一句话说明：** 不是又一篇新算法，而是一篇「实测对比」基准论文：作者用同一只「手提四传感器架」、同一组室内外轨迹，把四款最常被人形 / 移动机器人引用的商用闭源 VIO拉到同一条尺子上量——结论是 Apple ARKit 综合最稳最准（相对位姿误差 ≈ 0.02 m/s 漂移），但只能跑 iOS、对 ROS / Linux 不友好；T265 和 ZED 2 虽然 ROS 友好，但分别栽在「单目尺度漂移」和「旋转估计破坏正交性」上，给后续工程选型提供了一个可重复的硬证据。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-an-empirical-evaluation-of-four-off-the-shelf-pr](../../wiki/entities/paper-notebook-an-empirical-evaluation-of-four-off-the-shelf-pr.md).

## 对 wiki 的映射

- [paper-notebook-an-empirical-evaluation-of-four-off-the-shelf-pr](../../wiki/entities/paper-notebook-an-empirical-evaluation-of-four-off-the-shelf-pr.md)
- 分类父节点：[paper-notebook-category-09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems/An_Empirical_Evaluation_of_Four_Off-the-Shelf_Proprietary_VIO_Systems.html>
- 论文：<https://arxiv.org/abs/2207.06780>
