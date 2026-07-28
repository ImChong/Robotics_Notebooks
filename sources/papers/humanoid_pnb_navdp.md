# NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2505.08712>
- **项目页：** <https://wzcai99.github.io/navigation-diffusion-policy.github.io/>
- **代码：** <https://github.com/InternRobotics/NavDP>（**已开源**；README 标注 CC BY-NC-SA 4.0，checkpoint 需表单）
- **入库日期：** 2026-07-10
- **一句话说明：** 在动态复杂开放世界中导航是自主机器人的关键且困难的能力。已有方法多依赖级联模块化框架（需大量调参）或有限真实演示学习。NavDP（Navigation Diffusion Policy）是一个端到端网络，仅在仿真训练就能实现零样本 sim-to-real，跨多样环境与机器人本体迁移。它用统一的 Transformer 架构同时做轨迹生成与评估：以局部 RGB-D 观测为条件，为对比轨迹样本预测评论值（critic values），并借助特权仿真信息提升空间理解。训练数据大规模——跨 3000 个场景、累计超百万米导航。结果：在仿真与真机评测中均显著超越此前 SOTA。

## 核心摘录（策展，非全文）

- **数据：** ESDF + A* + BlenderProc 生成 1244 scenes、56k trajectories、10M RGB-D images、363.2 km。
- **方法：** 共享 Transformer 的 diffusion head 生成多条轨迹；goal-agnostic critic 用 ESDF 标注的旋转 / 插值负样本学安全排序。
- **结果：** PointGoal mSR/mSPL 70.4/58.6；Go2、Galaxea R1、G1 零样本真机；少量 real-to-sim 数据使目标场景 SR 50→80%。
- **开源边界：** 当前仓库可跑 `navdp_server.py` 与 IsaacSim benchmark，但原论文 checkpoint 需申请，仓库也已扩展到 InternVLA-N1。
- 知识归纳见 wiki 实体页：[paper-notebook-navdp-learning-sim-to-real-navigation-diffusion](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md).

## 对 wiki 的映射

- [paper-notebook-navdp-learning-sim-to-real-navigation-diffusion](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)
- [NavDP 项目页归档](../sites/navdp.md)
- [NavDP 仓库归档](../repos/navdp.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html>
- 论文：<https://arxiv.org/abs/2505.08712>
