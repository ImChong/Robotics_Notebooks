# TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2508.00355>
- **入库日期：** 2026-07-10
- **一句话说明：** 人形能做多样操作，前提是鲁棒精确的站立控制器。已有方法要么难精控高维上身关节、要么难同时保证鲁棒与精度——尤其当上身运动快时。本文提出一个新颖的时间优化策略（Time Optimization Policy, TOP），训练一个站立操作控制模型，同时保证平衡、精度与时间效率。核心思想是：调整上身动作的时间轨迹，而不只是一味强化下身的抗扰能力——让快速上身运动在时间上"错峰"，减轻对平衡的冲击。方法用 VAE 编码上身动作先验，并解耦全身控制（上身 PD 控制器 + 下身 RL 控制器）。仿真与真机实验表明，TOP 在站立操作上稳定且精确，优于已有方法。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-top-time-optimization-policy-for-stable-and-accu](../../wiki/entities/paper-notebook-top-time-optimization-policy-for-stable-and-accu.md).

## 对 wiki 的映射

- [paper-notebook-top-time-optimization-policy-for-stable-and-accu](../../wiki/entities/paper-notebook-top-time-optimization-policy-for-stable-and-accu.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation/TOP__Time_Optimization_Policy_for_Stable_and_Accurate_Standing_Manipulation.html>
- 论文：<https://arxiv.org/abs/2508.00355>
