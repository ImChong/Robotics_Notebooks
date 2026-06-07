# MAGNet: Diffusion Forcing for Multi-Agent Interaction Sequence Modeling

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** MAGNet: Diffusion Forcing for Multi-Agent Interaction Sequence Modeling
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2512.17900>
- **入库日期：** 2026-06-07
- **一句话说明：** 把 Diffusion Forcing（每个 token 独立加噪/独立去噪的自回归扩散）从单序列搬到多人交互——把每个人的姿态先用 VQ-VAE 压成 token，再把所有人的 token 交错喂给同一个 Transformer，训练时每个 token 独立采噪声、推理时按需控制每个人/每个时刻的噪声等级，从而一个模型同时支持双人/三人/N 人预测、Partner Inpainting、Partner Prediction、超长动作生成等任务。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-magnet](../../wiki/entities/paper-notebook-magnet.md).

## 对 wiki 的映射

- [paper-notebook-magnet](../../wiki/entities/paper-notebook-magnet.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling.html>
- 论文：<https://arxiv.org/abs/2512.17900>
