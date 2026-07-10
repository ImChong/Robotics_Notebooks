# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MANIKIN__Biomechanically_Accurate_Neural_Inverse_Kinematics_for_Human_Motion_Estimation/MANIKIN__Biomechanically_Accurate_Neural_Inverse_Kinematics_for_Human_Motion_Estimation.html>
- **分类：** 14_Human_Motion
- **入库日期：** 2026-07-10
- **一句话说明：** 混合现实（MR）系统常需仅从末端（主要是头与手）位姿估计用户全身关节配置——即从稀疏观测解逆运动学（IK）得全身骨架。但现有方法沿运动链累积误差，导致预测末端与输入位姿不对齐（手位偏差、脚穿地等）。MANIKIN 是一个神经-解析（neural-analytic）IK 求解器，仅用头与手位姿即可跟踪全身动作。其关键是：精炼常用的 SMPL 参数模型，嵌入解剖学约束、缩减特定参数的自由度以更贴近人体生物力学，确保物理可信的姿态预测；并基于摆转角（swivel angle）预测，使输出完美匹配输入末端位姿、同时避免地面穿插。方法在快速推理下，于定量与定性上超越 SOTA（ECCV 2024）。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-manikin-biomechanically-accurate-neural-inverse](../../wiki/entities/paper-notebook-manikin-biomechanically-accurate-neural-inverse.md).

## 对 wiki 的映射

- [paper-notebook-manikin-biomechanically-accurate-neural-inverse](../../wiki/entities/paper-notebook-manikin-biomechanically-accurate-neural-inverse.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MANIKIN__Biomechanically_Accurate_Neural_Inverse_Kinematics_for_Human_Motion_Estimation/MANIKIN__Biomechanically_Accurate_Neural_Inverse_Kinematics_for_Human_Motion_Estimation.html>

