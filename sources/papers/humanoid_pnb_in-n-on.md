# In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2511.15704>
- **入库日期：** 2026-07-10
- **一句话说明：** 第一视角（egocentric）视频是学操作策略的宝贵可扩展数据源，但数据异质性大，多数方法只把人类数据用于简单预训练，没释放全部潜力。本文先给出一套可扩展配方：把人类数据分成两类——野外（in-the-wild）与任务对齐（on-task），并系统分析如何使用。作者整理出数据集 PHSD，含 1000+ 小时多样野外第一视角数据与 20+ 小时直接对齐目标任务的任务数据。据此训练一个大型语言条件流匹配策略 Human0；配合域适应技术，Human0 缩小人到人形的差距。实证表明，规模化人类数据带来若干新性质：仅凭人类数据就能听从语言指令、少样本学习、以及用任务数据提升的鲁棒性。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-in-n-on-scaling-egocentric-manipulation-with-in](../../wiki/entities/paper-notebook-in-n-on-scaling-egocentric-manipulation-with-in.md).

## 对 wiki 的映射

- [paper-notebook-in-n-on-scaling-egocentric-manipulation-with-in](../../wiki/entities/paper-notebook-in-n-on-scaling-egocentric-manipulation-with-in.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html>
- 论文：<https://arxiv.org/abs/2511.15704>
