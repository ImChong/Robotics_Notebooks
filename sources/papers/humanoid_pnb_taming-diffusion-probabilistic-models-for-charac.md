# Taming Diffusion Probabilistic Models for Character Control

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Taming Diffusion Probabilistic Models for Character Control
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Taming_Diffusion_Probabilistic_Models_for_Character_Control/Taming_Diffusion_Probabilistic_Models_for_Character_Control.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2404.15121>
- **入库日期：** 2026-07-10
- **一句话说明：** 本文提出一个新颖的角色控制框架，有效利用动作扩散概率模型生成高质量、多样的角色动画，并实时响应各种动态用户控制信号。系统核心是一个基于 Transformer 的条件自回归动作扩散模型（Conditional Autoregressive Motion Diffusion Model, CAMDM），依据历史动作与粗粒度用户控制生成可能的未来动作。为实现高质量实时控制，框架用三招：条件单独 token 化、对历史动作的无分类器引导（classifier-free guidance）、以及启发式未来轨迹外延（提升计算效率）。这是首个支持基于用户交互控制实时生成高质量多样角色动画的模型，单一统一模型支持多种动画风格与多样行走技能。SIGGRAPH 2024。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-taming-diffusion-probabilistic-models-for-charac](../../wiki/entities/paper-notebook-taming-diffusion-probabilistic-models-for-charac.md).

## 对 wiki 的映射

- [paper-notebook-taming-diffusion-probabilistic-models-for-charac](../../wiki/entities/paper-notebook-taming-diffusion-probabilistic-models-for-charac.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Taming_Diffusion_Probabilistic_Models_for_Character_Control/Taming_Diffusion_Probabilistic_Models_for_Character_Control.html>
- 论文：<https://arxiv.org/abs/2404.15121>
