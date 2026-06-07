# InEKFormer: A Hybrid State Estimator for Humanoid Robots

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** InEKFormer: A Hybrid State Estimator for Humanoid Robots
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots.html>
- **分类：** 09_State_Estimation
- **arXiv：** <https://arxiv.org/abs/2511.16306>
- **入库日期：** 2026-06-07
- **一句话说明：** InEKFormer 把经典 不变扩展卡尔曼滤波（InEKF） 的几何结构保留下来，但让 Transformer 从一段「状态 / 观测残差」的历史里隐式输出噪声相关的修正量，从而绕开「手调噪声协方差 Q/R」这件让所有滤波工程师头大的活；在 RH5 真机数据上跟 InEKF / KalmanNet 两条基线对照，验证了 Transformer 在人形高维状态估计里的可行性，同时也点出了「自回归训练不鲁棒就会爆」的现实问题。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-inekformer](../../wiki/entities/paper-notebook-inekformer.md).

## 对 wiki 的映射

- [paper-notebook-inekformer](../../wiki/entities/paper-notebook-inekformer.md)
- 分类父节点：[paper-notebook-category-09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots/InEKFormer__A_Hybrid_State_Estimator_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2511.16306>
