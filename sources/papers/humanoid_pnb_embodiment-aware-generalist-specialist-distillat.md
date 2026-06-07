# Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2602.02960>
- **入库日期：** 2026-06-07
- **一句话说明：** EAGLE 把"跨本体人形 WBC"建成一个迭代的"泛化—专家"蒸馏循环：先在一个池子里同时训练多种本体的泛化策略；再为每个本体派生一个专家做精修；最后把各专家的新技能通过 DAgger 蒸馏回泛化策略，反复循环直至收敛——配合一套统一的高维指令接口（蹲、倾、底盘速度等同时支持），最终用一份策略驱动 H1 / G1 / N1 / T1 / Adam 等异构人形。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-embodiment-aware-generalist-specialist-distillat](../../wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md).

## 对 wiki 的映射

- [paper-notebook-embodiment-aware-generalist-specialist-distillat](../../wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B.html>
- 论文：<https://arxiv.org/abs/2602.02960>
