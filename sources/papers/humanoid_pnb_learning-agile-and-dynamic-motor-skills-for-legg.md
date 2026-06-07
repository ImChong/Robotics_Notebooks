# Learning Agile and Dynamic Motor Skills for Legged Robots

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Learning Agile and Dynamic Motor Skills for Legged Robots
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Learning_Agile_and_Dynamic_Motor_Skills_for_Legged_Robots/Learning_Agile_and_Dynamic_Motor_Skills_for_Legged_Robots.html>
- **分类：** 03_High_Impact_Selection
- **子分类：** 仿真到现实与基座模型
- **arXiv：** <https://arxiv.org/abs/1901.08652>
- **入库日期：** 2026-06-07
- **一句话说明：** 把"电机 + 减速箱 + 控制器 + 通信延迟"全部用一个 LSTM 致动器网络（actuator network） 离线辨识，然后在 RaiSim 里以神经网络代替传统刚体力学做高速 RL 训练，最终把策略零样本搬到 ANYmal 上，让它能跟随速度指令、奔跑（最高 1.5 m/s，比厂家 MPC 快 25%）以及从任意倒地姿态自主翻身爬起——首次系统性证明 sim-to-real RL 可以在真实四足上稳定落地。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg](../../wiki/entities/paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg.md).

## 对 wiki 的映射

- [paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg](../../wiki/entities/paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Learning_Agile_and_Dynamic_Motor_Skills_for_Legged_Robots/Learning_Agile_and_Dynamic_Motor_Skills_for_Legged_Robots.html>
- 论文：<https://arxiv.org/abs/1901.08652>
