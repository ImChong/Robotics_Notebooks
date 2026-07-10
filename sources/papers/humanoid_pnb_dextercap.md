# DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2601.05844>
- **入库日期：** 2026-07-10
- **一句话说明：** 精细的「在手」灵巧操作很难采集：手指挨得很近导致严重自遮挡，且动作幅度细微，传统光学动捕要么相机昂贵、要么后处理人工成本巨大。DexterCap 用密集的「字符编码」标记贴片（高对比棋盘格，每格带唯一双字符 ID）贴满手部各刚性区域，配合三级（marker → edge → tag）检测识别模型在自遮挡下稳定追踪，再用自动化重建流水线把 3D 标记拟合到 MANO 手模型与物体模型，恢复逐帧手参数与物体位姿/铰接状态——低成本、少人工地采到从简单基元到魔方等复杂铰接物的精细手-物交互，并发布 DexterHand 数据集与代码。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-dextercap](../../wiki/entities/paper-notebook-dextercap.md).

## 对 wiki 的映射

- [paper-notebook-dextercap](../../wiki/entities/paper-notebook-dextercap.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object.html>
- 论文：<https://arxiv.org/abs/2601.05844>
