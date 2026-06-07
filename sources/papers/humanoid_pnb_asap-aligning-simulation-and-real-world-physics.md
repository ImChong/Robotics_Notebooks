# ASAP Aligning Simulation and Real-World Physics for Agile Humanoid Skills

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** ASAP Aligning Simulation and Real-World Physics for Agile Humanoid Skills
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills.html>
- **分类：** 03_High_Impact_Selection
- **子分类：** 仿真到现实与基座模型
- **arXiv：** <https://arxiv.org/abs/2502.01143>
- **入库日期：** 2026-06-07
- **一句话说明：** 先用人类视频重定向后的参考动作在仿真里预训练运动跟踪策略，再在真机 rollout 收集状态轨迹，用残差（delta）动作模型显式补偿仿真与真机的动力学差；把该模型冻结后嵌入仿真器做「物理对齐」式的策略微调，最后在真机去掉 delta 模型直接部署——在侧跳、前跳、踢球、球星庆祝动作等全身敏捷技能上显著降低跟踪误差，并优于纯 SysID、纯域随机化、以及仅学习 delta 动力学但不回灌仿真的基线。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-asap-aligning-simulation-and-real-world-physics](../../wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md).

## 对 wiki 的映射

- [paper-notebook-asap-aligning-simulation-and-real-world-physics](../../wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills.html>
- 论文：<https://arxiv.org/abs/2502.01143>
