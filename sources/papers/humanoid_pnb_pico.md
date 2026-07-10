# PICO: Reconstructing 3D People In Contact with Objects

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** PICO: Reconstructing 3D People In Contact with Objects
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PICO__Reconstructing_3D_People_In_Contact_with_Objects/PICO__Reconstructing_3D_People_In_Contact_with_Objects.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2504.17695>
- **入库日期：** 2026-07-10
- **一句话说明：** 从单张彩色图恢复 3D 人-物交互（HOI）很难：深度歧义、遮挡、物体形状外观差异巨大。以往工作需受控设置（已知物体形状/接触）且只处理有限物体类。PICO 想泛化到自然图像与新物体类，用两条思路：① 构建 PICO-db ——自然图像，唯一地配对身体与物体网格上的稠密 3D 接触：借视觉基础模型从数据库检索合适 3D 物体网格，再用一种每补丁仅 2 次点击的新方法把（DAMON 的）身体接触补丁投影到物体，以最小人工建立丰富的身-物接触对应；② 用 PICO-fit ——一种渲染-比较（render-and-compare）拟合方法，为 SMPL-X 身体推断接触、从 PICO-db 检索可能的 3D 物体网格与接触，并据接触迭代拟合身体与物体网格到图像证据。PICO 对许多现有方法无法处理的物体类别都работает（泛化好）。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-pico-reconstructing-3d-people-in-contact-with-ob](../../wiki/entities/paper-notebook-pico-reconstructing-3d-people-in-contact-with-ob.md).

## 对 wiki 的映射

- [paper-notebook-pico-reconstructing-3d-people-in-contact-with-ob](../../wiki/entities/paper-notebook-pico-reconstructing-3d-people-in-contact-with-ob.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PICO__Reconstructing_3D_People_In_Contact_with_Objects/PICO__Reconstructing_3D_People_In_Contact_with_Objects.html>
- 论文：<https://arxiv.org/abs/2504.17695>
