# AutoOdom: Learning Auto-regressive Proprioceptive Odometry for Legged Locomotion

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** AutoOdom: Learning Auto-regressive Proprioceptive Odometry for Legged Locomotion
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/AutoOdom__Learning_Auto-regressive_Proprioceptive_Odometry_for_Legged_Locomotio/AutoOdom__Learning_Auto-regressive_Proprioceptive_Odometry_for_Legged_Locomotio.html>
- **分类：** 09_State_Estimation
- **arXiv：** <https://arxiv.org/abs/2511.18857>
- **入库日期：** 2026-06-07
- **一句话说明：** AutoOdom 把"足式机器人本体感知里程计（只用 IMU + 关节传感器）"这件事纯学习化：第一阶段在大规模仿真里学到非线性动力学和频繁变化的接触状态，第二阶段在少量真机数据上做自回归微调——让模型学着"喂自己的预测当输入"，由此自然抑制传感器噪声和累计漂移，在 Booster T1 上把 ATE / RPE 相比 Legolas 砍掉了 36%–59%。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-autoodom](../../wiki/entities/paper-notebook-autoodom.md).

## 对 wiki 的映射

- [paper-notebook-autoodom](../../wiki/entities/paper-notebook-autoodom.md)
- 分类父节点：[paper-notebook-category-09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/AutoOdom__Learning_Auto-regressive_Proprioceptive_Odometry_for_Legged_Locomotio/AutoOdom__Learning_Auto-regressive_Proprioceptive_Odometry_for_Legged_Locomotio.html>
- 论文：<https://arxiv.org/abs/2511.18857>
