# Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2508.12252>
- **入库日期：** 2026-07-10
- **一句话说明：** 人形机器人「直接在真机上做 RL」一直难落地：怕摔坏、奖励难设计、训练效率低、还得有人全程看着。RTR 的核心点子是再加一台机器人当「老师」——用一台 UR5 机械臂在训练全程托举/保护人形、按课程逐步放手、施加扰动、检测失败、自动复位，把原本需要人手的环节全自动化，从而支持长时间、低人工监督的真机训练。配套提出一套 sim-to-real 流程：先在仿真里训一个把动力学编码进单个隐变量的策略，再在真机上只微调这个隐变量 + 重训 critic，实现高效适配。两个真机任务验证：把行走策略微调到精确速度跟踪、以及从零学会荡秋千式摆动。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-robot-trains-robot-automatic-real-world-policy-a](../../wiki/entities/paper-notebook-robot-trains-robot-automatic-real-world-policy-a.md).

## 对 wiki 的映射

- [paper-notebook-robot-trains-robot-automatic-real-world-policy-a](../../wiki/entities/paper-notebook-robot-trains-robot-automatic-real-world-policy-a.md)
- 分类父节点：[paper-notebook-category-10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html>
- 论文：<https://arxiv.org/abs/2508.12252>
