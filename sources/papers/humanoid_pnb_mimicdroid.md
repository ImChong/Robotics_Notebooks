# MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2509.09769>
- **入库日期：** 2026-07-10
- **一句话说明：** 目标是让人形从少量视频示例高效解决新操作任务。上下文学习（ICL）因测试时数据高效、快速适应而有前景，但现有 ICL 方法依赖费力的遥操作数据，难规模化。本文用人类玩耍视频（human play videos）——人们自由与环境交互的连续、无标注视频——作为可扩展、多样的训练源。提出 MimicDroid：仅用人类玩耍视频做训练，抽取行为相似的轨迹对，训练策略以一条轨迹为条件预测另一条的动作，从而获得测试时适应新物体/环境的 ICL 能力。为弥合具身差距，先用运动学相似性把 RGB 视频估计的人手腕姿态重定向到人形；训练时随机块遮挡（patch masking）降低对人类特有线索的过拟合、增强对视觉差异的鲁棒。作者还提出一个开源仿真基准（难度递增）评估少样本学习；MimicDroid 优于 SOTA，真机成功率近两倍。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo](../../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md).

## 对 wiki 的映射

- [paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo](../../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html>
- 论文：<https://arxiv.org/abs/2509.09769>
