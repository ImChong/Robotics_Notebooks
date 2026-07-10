# The Invariant Extended Kalman Filter as a Stable Observer

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** The Invariant Extended Kalman Filter as a Stable Observer
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer.html>
- **分类：** 09_State_Estimation
- **arXiv：** <https://arxiv.org/abs/1410.1465>
- **入库日期：** 2026-07-10
- **一句话说明：** 标准 EKF 在非线性系统上把动力学「就地线性化」，线性化质量取决于当前估计，估计一偏、线性化就坏、滤波就可能发散。本文从确定性观测器的视角重新审视 不变 EKF（InEKF）：作者定义了一类「群仿射（group-affine）」系统，证明在这类系统上，建立在李群上的估计误差演化是自治的（不依赖具体轨迹/估计），并且满足精确的对数线性微分方程；由此推出 InEKF 在标准可观测性条件下局部渐近收敛——也就是说它是一个有理论保证的「稳定观测器」，而经典 EKF 没有这种保证。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-the-invariant-extended-kalman-filter-as-a-stable](../../wiki/entities/paper-notebook-the-invariant-extended-kalman-filter-as-a-stable.md).

## 对 wiki 的映射

- [paper-notebook-the-invariant-extended-kalman-filter-as-a-stable](../../wiki/entities/paper-notebook-the-invariant-extended-kalman-filter-as-a-stable.md)
- 分类父节点：[paper-notebook-category-09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer/The_Invariant_Extended_Kalman_Filter_as_a_Stable_Observer.html>
- 论文：<https://arxiv.org/abs/1410.1465>
