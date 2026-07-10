# Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2507.15597>
- **入库日期：** 2026-07-10
- **一句话说明：** Being-H0 是一个在大规模人类视频上训练的灵巧视觉-语言-动作模型（VLA）。现有 VLA 在高灵巧操作上吃力、对新场景泛化差，主因是依赖有 sim-to-real 差距的合成数据或缺规模与多样性的遥操作演示。为破数据瓶颈，本文把人手当作基础操作器（foundation manipulator），利用网络数据中丰富的灵巧性与可扩展性。方法核心是物理指令微调（physical instruction tuning）：结合大规模人类视频 VLA 预训练、3D 推理的物理空间对齐、以及面向机器人任务的后训练适配。还提出部件级运动 token 化（part-level motion tokenization），达毫米级重建精度以建模精确手部轨迹；并构建融合动捕、VR、RGB-only 视频的百万级运动指令数据集。实验显示 Being-H0 在手部动作生成与指令跟随上优异，随模型与数据规模良好扩展，并在真机操作上随物理指令微调见效。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-being-h0-vision-language-action-pretraining-from](../../wiki/entities/paper-notebook-being-h0-vision-language-action-pretraining-from.md).

## 对 wiki 的映射

- [paper-notebook-being-h0-vision-language-action-pretraining-from](../../wiki/entities/paper-notebook-being-h0-vision-language-action-pretraining-from.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos.html>
- 论文：<https://arxiv.org/abs/2507.15597>
