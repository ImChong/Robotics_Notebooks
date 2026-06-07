# Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation.html>
- **分类：** 09_State_Estimation
- **arXiv：** <https://arxiv.org/abs/2507.10105>
- **入库日期：** 2026-06-07
- **一句话说明：** 要让一台没有关节力矩传感器的人形机器人也能做力矩控制，就必须把"力矩"从其他传感器里估出来；论文的做法是：先用 PINN 把谐波减速器最难刻画的非线性摩擦学下来，再把 PINN 的摩擦估计当作 UKF 的一个测量量喂进去，最终在 ergoCub 真机平衡实验上让腿部 6 个关节的力矩跟踪 RMSE 落到 0.08–1.41 Nm，整体优于工业界默认基线 RNEA。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-physics-informed-neural-networks-with-ukf-for-se](../../wiki/entities/paper-notebook-physics-informed-neural-networks-with-ukf-for-se.md).

## 对 wiki 的映射

- [paper-notebook-physics-informed-neural-networks-with-ukf-for-se](../../wiki/entities/paper-notebook-physics-informed-neural-networks-with-ukf-for-se.md)
- 分类父节点：[paper-notebook-category-09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation/Physics-Informed_Neural_Networks_with_UKF_for_Sensorless_Joint_Torque_Estimation.html>
- 论文：<https://arxiv.org/abs/2507.10105>
