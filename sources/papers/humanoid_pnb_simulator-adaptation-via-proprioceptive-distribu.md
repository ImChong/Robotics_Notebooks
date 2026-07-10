# Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2604.11090>
- **入库日期：** 2026-07-10
- **一句话说明：** 仿真训出来的腿足策略一上真机就掉点，根源是仿真与真实动力学有偏差。常见做法是去改策略（域随机化、在线适配），本文反其道：去改仿真器——让仿真更像真机，再在校准后的仿真里训策略就能直接迁移。难点在于「怎么衡量仿真像不像真机」：传统做法要逐时刻对齐轨迹，依赖动捕/特权传感、对时间对齐敏感。本文提出本体感知分布匹配（Proprioceptive Distribution Matching）：把真机与仿真各自跑一段，只看「关节观测 + 动作」的统计分布像不像，无需时间对齐、无需外部传感。用黑盒优化在这个分布距离上辨识仿真参数（或学习 action-delta / 残差执行器模型），不到 5 分钟真机数据就能显著降低漂移，效果可比肩用特权状态对齐的基线。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-simulator-adaptation-via-proprioceptive-distribu](../../wiki/entities/paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md).

## 对 wiki 的映射

- [paper-notebook-simulator-adaptation-via-proprioceptive-distribu](../../wiki/entities/paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md)
- 分类父节点：[paper-notebook-category-10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching.html>
- 论文：<https://arxiv.org/abs/2604.11090>
