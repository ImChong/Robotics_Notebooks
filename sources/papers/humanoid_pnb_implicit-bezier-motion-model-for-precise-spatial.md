# Implicit Bézier Motion Model for Precise Spatial and Temporal Control

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Implicit Bézier Motion Model for Precise Spatial and Temporal Control
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control.html>
- **分类：** 14_Human_Motion
- **入库日期：** 2026-07-10
- **一句话说明：** 隐式贝塞尔运动模型（Implicit Bézier Motion Model, IBMM）提供对生成运动的细粒度空间与时间控制。它针对此前贝塞尔运动模型（BMM）的关键局限——BMM 只能在均匀时间间隔预测一组固定控制点，使艺术家无法做细粒度时间控制（如在时间上移动控制点、或在需要更多细节的区域增加控制点）。IBMM 在训练时隐式学习贝塞尔拟合，支持任意时间控制点，无需对数据预先拟合，并彻底取消「步幅（stride）」概念，使艺术家可在任意帧约束任意末端关节。此外，IBMM 还为用户引入一项新的全局控制：对运动全局缓入/缓出（ease-in/out）的直接手柄——这是首个在生成自然运动时无需人工标注即可全局控制时间的方法。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-implicit-b-zier-motion-model-for-precise-spatial](../../wiki/entities/paper-notebook-implicit-b-zier-motion-model-for-precise-spatial.md).

## 对 wiki 的映射

- [paper-notebook-implicit-b-zier-motion-model-for-precise-spatial](../../wiki/entities/paper-notebook-implicit-b-zier-motion-model-for-precise-spatial.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control/Implicit_Bezier_Motion_Model_for_Precise_Spatial_and_Temporal_Control.html>

