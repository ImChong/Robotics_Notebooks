# Masquerade: Learning from In-the-wild Human Videos using Data-Editing

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Masquerade: Learning from In-the-wild Human Videos using Data-Editing
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2508.09976>
- **入库日期：** 2026-07-10
- **一句话说明：** 机器人操作仍数据稀缺——最大的机器人数据集也比驱动语言/视觉突破的数据小几个数量级。Masquerade 通过编辑野外第一视角人类视频来闭合人-机视觉具身差距，再用编辑后的视频学机器人策略。流程把每段人类视频变成机器人化演示：① 估计 3D 手姿；② 修复涂抹（inpaint）人臂；③ 叠加渲染的双臂机器人，使其跟踪恢复的末端轨迹。在 67.5 万帧编辑片段上预训练视觉编码器以预测未来 2D 机器人关键点，并在每任务仅 50 条机器人演示上微调扩散策略头（继续保留该辅助损失），所得策略泛化显著更好。在三个长时程双手厨房任务、各三个未见场景上，Masquerade 较基线高 5–6 倍；消融显示机器人叠加与协同训练都不可或缺，性能随编辑人类视频量对数增长。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-masquerade-learning-from-in-the-wild-human-video](../../wiki/entities/paper-notebook-masquerade-learning-from-in-the-wild-human-video.md).

## 对 wiki 的映射

- [paper-notebook-masquerade-learning-from-in-the-wild-human-video](../../wiki/entities/paper-notebook-masquerade-learning-from-in-the-wild-human-video.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing.html>
- 论文：<https://arxiv.org/abs/2508.09976>
