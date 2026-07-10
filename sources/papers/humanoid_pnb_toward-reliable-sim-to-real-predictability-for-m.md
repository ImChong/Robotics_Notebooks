# Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion.html>
- **分类：** 05_Locomotion
- **arXiv：** <https://arxiv.org/abs/2602.00678>
- **入库日期：** 2026-07-10
- **一句话说明：** RL 在四足敏捷运动上很有前景，即便仅本体感受也行。但实践中sim-to-real 差距与复杂地形上的奖励过拟合会让策略迁移失败，而物理验证又风险高、低效。本文提出一个统一框架：① 一个专家混合（MoE）运动策略，用门控的专家集合把隐式地形与指令建模分解，仅靠本体感受实现更优的部署鲁棒性与泛化；② RoboGauge ——一个预测性评估套件，量化 sim-to-real 可迁移性，通过跨地形、难度、域随机化的sim-to-sim 测试给出多维本体感受指标，使无需大量真机试验即可可靠地选 MoE 策略。在 Unitree Go2 上：雪、沙、楼梯、斜坡、30cm 障碍等未见地形稳健通行；高速测试达 4 m/s，并涌现与高速稳定相关的窄步态。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-toward-reliable-sim-to-real-predictability-for-m](../../wiki/entities/paper-notebook-toward-reliable-sim-to-real-predictability-for-m.md).

## 对 wiki 的映射

- [paper-notebook-toward-reliable-sim-to-real-predictability-for-m](../../wiki/entities/paper-notebook-toward-reliable-sim-to-real-predictability-for-m.md)
- 分类父节点：[paper-notebook-category-05-locomotion](../../wiki/overview/paper-notebook-category-05-locomotion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion/Toward_Reliable_Sim-to-Real_Predictability_for_MoE-based_Robust_Quadrupedal_Locomotion.html>
- 论文：<https://arxiv.org/abs/2602.00678>
