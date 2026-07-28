# NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration

> 来源归档（ingest · arXiv / 官方项目 / Humanoid Paper Notebooks progress）

- **标题：** NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration
- **类型：** paper
- **深读状态：** 待撰写（见 [papers/PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)）
- **计划笔记路径：** `papers/08_Navigation/nomad-goal-masked-diffusion-policies-for-navigat/nomad-goal-masked-diffusion-policies-for-navigat.md`
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2310.07896>
- **项目页：** <https://general-navigation-models.github.io/nomad/>
- **代码：** <https://github.com/robodhruv/visualnav-transformer>（**已开源**，MIT；含 checkpoint、训练与 ROS 部署）
- **入库日期：** 2026-06-11
- **一句话说明：** NoMaD 用 attention goal mask 在同一扩散策略中统一 ImageGoal 导航与无目标探索，并以拓扑记忆完成长程任务。

## 核心摘录（策展，非全文）

- **架构：** EfficientNet-B0 + 4-layer Transformer；goal mask 以 0.5 概率屏蔽目标；15-layer 1D UNet 经 10 步扩散生成未来动作。
- **数据：** GNM / SACSoN 等多机器人真实 RGB 轨迹 100+ h；公开 checkpoint 可用，但部分训练数据不公开。
- **结果：** 6 个真实室内外环境；探索相对 subgoal diffusion 提升 25%+，参数约少 15×；与拓扑 graph 配合做 goal discovery / navigation。
- **开源：** `train/train.py`、`navigate.sh`、`explore.sh` 与权重均发布；环境基于 ROS Noetic / 较旧 CUDA。
- 知识归纳见 wiki 实体页：[paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md).

## 对 wiki 的映射

- [paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)
- [NoMaD 项目页归档](../sites/nomad.md)
- [visualnav-transformer 仓库归档](../repos/visualnav-transformer-nomad.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- [Humanoid Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2310.07896>
