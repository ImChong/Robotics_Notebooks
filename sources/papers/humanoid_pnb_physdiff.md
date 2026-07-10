# PhysDiff: Physics-Guided Human Motion Diffusion Model

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** PhysDiff: Physics-Guided Human Motion Diffusion Model
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2212.02500>
- **入库日期：** 2026-07-10
- **一句话说明：** 去噪扩散模型在人体动作生成上效果很好，但现有动作扩散模型忽视物理定律，常生成带明显伪影的动作——漂浮（floating）、脚滑（foot sliding/skating）、地面穿插（ground penetration）等。PhysDiff 把物理约束注入扩散过程：提出一个物理引导的动作投影（physics-guided motion projection）模块——在扩散的去噪步中，借物理仿真器里的动作模仿（motion imitation），把当前扩散出的（含噪）动作投影成一个物理可行的动作，再用它引导下一步去噪。如此生成的动作物理可信、自然，大幅减少上述伪影，在大规模人体动作数据集上取得SOTA 的动作质量与物理可信度。ICCV 2023 Oral。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-physdiff-physics-guided-human-motion-diffusion-m](../../wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md).

## 对 wiki 的映射

- [paper-notebook-physdiff-physics-guided-human-motion-diffusion-m](../../wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model/PhysDiff__Physics-Guided_Human_Motion_Diffusion_Model.html>
- 论文：<https://arxiv.org/abs/2212.02500>
