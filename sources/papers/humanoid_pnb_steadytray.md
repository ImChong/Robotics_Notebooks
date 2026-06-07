# SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2603.10306>
- **入库日期：** 2026-06-07
- **一句话说明：** SteadyTray 把"端托盘 + 走路"这件高耦合的活，显式拆成两层 RL：底层用一个稳健的人形行走策略当老师，上层挂一个残差模块专门抵消步态引起的末端抖动；通过四阶段课程（预训练 → 托盘微调 → 残差教师 → 学生蒸馏），在 Unitree G1 上做到 96.9% 速度跟踪成功率 / 74.5% 抗扰鲁棒性，并且零样本 sim-to-real 落地真机。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md).

## 对 wiki 的映射

- [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid/SteadyTray__Learning_Object_Balancing_Tasks_in_Humanoid_Tray_Transport_via_Resid.html>
- 论文：<https://arxiv.org/abs/2603.10306>
