# Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2504.06585>
- **入库日期：** 2026-07-10
- **一句话说明：** 把策略从仿真搬到真机失败，根子在现实差（reality gap）——仿真没建模的非线性执行器动力学、关节柔顺、接触顺应等。主流做法是域随机化（DR）：把摩擦、质量、电机常数等十几个参数在区间里乱抽。但 DR 的表达力被「参数化」框死——它只能在预先选定的参数维度上抖动，抖不出那些状态相关、非参数化的复杂偏差。本文换个空间下手：在关节力矩上直接加一个状态相关的扰动项 τ_φ(s)，由一张随机初始化、训练全程不更新的小 MLP 生成，每个 episode 重抽一次权重。这样注入的扰动天然依赖机器人当前状态（姿态、速度、接触力……），能逼真模拟「在某些位形下执行器更软、某些接触下力更偏」这类 DR 给不出的差异，从而把策略训得对没见过的现实差更鲁棒。仿真里遇到未见的执行器刚度和软地面，DR/ERFI 全军覆没而本法照走；真机 TOCABI 上 3/3 成功步行，DR 2/3、ERFI 0/3。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-sim-to-real-of-humanoid-locomotion-policies-via](../../wiki/entities/paper-notebook-sim-to-real-of-humanoid-locomotion-policies-via.md).

## 对 wiki 的映射

- [paper-notebook-sim-to-real-of-humanoid-locomotion-policies-via](../../wiki/entities/paper-notebook-sim-to-real-of-humanoid-locomotion-policies-via.md)
- 分类父节点：[paper-notebook-category-10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection.html>
- 论文：<https://arxiv.org/abs/2504.06585>
