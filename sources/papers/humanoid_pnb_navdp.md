# NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2505.08712>
- **入库日期：** 2026-07-10
- **一句话说明：** 在动态复杂开放世界中导航是自主机器人的关键且困难的能力。已有方法多依赖级联模块化框架（需大量调参）或有限真实演示学习。NavDP（Navigation Diffusion Policy）是一个端到端网络，仅在仿真训练就能实现零样本 sim-to-real，跨多样环境与机器人本体迁移。它用统一的 Transformer 架构同时做轨迹生成与评估：以局部 RGB-D 观测为条件，为对比轨迹样本预测评论值（critic values），并借助特权仿真信息提升空间理解。训练数据大规模——跨 3000 个场景、累计超百万米导航。结果：在仿真与真机评测中均显著超越此前 SOTA。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-navdp-learning-sim-to-real-navigation-diffusion](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md).

## 对 wiki 的映射

- [paper-notebook-navdp-learning-sim-to-real-navigation-diffusion](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html>
- 论文：<https://arxiv.org/abs/2505.08712>
