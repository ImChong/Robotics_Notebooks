# DiffCoTune: Differentiable Co-Tuning for Cross-domain Robot Control

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** DiffCoTune: Differentiable Co-Tuning for Cross-domain Robot Control
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2505.24068>
- **入库日期：** 2026-07-10
- **一句话说明：** 机器人控制器部署常受建模差异所困——为可计算而简化模型、或仿真器本身不准——通常需要临时手工调参才能在目标域达标。DiffCoTune 提出一个自动、基于梯度的调参框架，借助可微仿真器（differentiable simulators）提升部署域性能。方法迭代地采集 rollout，协同调（co-tune）仿真器参数与控制器参数，使迁移能在目标域少数几次试验内系统完成。具体地，构造多步目标并用交替优化有效把控制器适配到部署域。框架的可扩展性体现在：能对任意复杂度的模型法与学习法控制器协同调参，任务从低维倒立摆到高维四足与双足跟踪，在不同部署域均见性能提升。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-diffcotune-differentiable-co-tuning-for-cross-do](../../wiki/entities/paper-notebook-diffcotune-differentiable-co-tuning-for-cross-do.md).

## 对 wiki 的映射

- [paper-notebook-diffcotune-differentiable-co-tuning-for-cross-do](../../wiki/entities/paper-notebook-diffcotune-differentiable-co-tuning-for-cross-do.md)
- 分类父节点：[paper-notebook-category-10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control.html>
- 论文：<https://arxiv.org/abs/2505.24068>
