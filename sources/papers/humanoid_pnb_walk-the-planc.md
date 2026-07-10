# Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds.html>
- **分类：** 05_Locomotion
- **arXiv：** <https://arxiv.org/abs/2601.06286>
- **入库日期：** 2026-07-10
- **一句话说明：** 踏脚石/稀疏落脚点上的人形行走，最难的是「敏捷」和「精准落脚」要同时满足：纯 model-free RL 在这种离散、受约束地形上很难学，常常退化成原地站着不动；纯模型法（落脚规划）落脚精准但动作保守、对未建模动力学不鲁棒。PLANC 把两者缝起来——用一个 降阶 LIP 落脚规划器实时生成「动力学一致」的全状态参考轨迹，再用 控制李雅普诺夫函数（CLF）奖励把 RL 策略引导到这条物理可行的参考上，最终在 Unitree G1 上实现既快又准、可真机部署的踏脚石行走。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-walk-the-planc-physics-guided-rl-for-agile-human](../../wiki/entities/paper-notebook-walk-the-planc-physics-guided-rl-for-agile-human.md).

## 对 wiki 的映射

- [paper-notebook-walk-the-planc-physics-guided-rl-for-agile-human](../../wiki/entities/paper-notebook-walk-the-planc-physics-guided-rl-for-agile-human.md)
- 分类父节点：[paper-notebook-category-05-locomotion](../../wiki/overview/paper-notebook-category-05-locomotion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds.html>
- 论文：<https://arxiv.org/abs/2601.06286>
