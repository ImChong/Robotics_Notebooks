# OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2410.11792>
- **入库日期：** 2026-07-10
- **一句话说明：** 研究从单段视频演示模仿来教人形机器人操作技能。OKAMI 从单段 RGB-D 视频生成操作计划并导出可执行策略。其核心是物体感知重定向（object-aware retargeting）：让人形复现视频中的人类动作，同时在部署时适应不同物体位置。OKAMI 用开放世界视觉模型识别任务相关物体，并分别重定向身体动作与手部姿态。实验表明 OKAMI 在多变视觉与空间条件下强泛化，在开放世界从观察模仿（imitation from observation）上超越 SOTA 基线。进一步地，用 OKAMI 的 rollout 轨迹训练闭环视觉运动策略，在无需费力遥操作的情况下达平均 79.2% 成功率。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-okami-teaching-humanoid-robots-manipulation-skil](../../wiki/entities/paper-notebook-okami-teaching-humanoid-robots-manipulation-skil.md).

## 对 wiki 的映射

- [paper-notebook-okami-teaching-humanoid-robots-manipulation-skil](../../wiki/entities/paper-notebook-okami-teaching-humanoid-robots-manipulation-skil.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation/OKAMI__Teaching_Humanoid_Robots_Manipulation_Skills_through_Single_Video_Imitation.html>
- 论文：<https://arxiv.org/abs/2410.11792>
