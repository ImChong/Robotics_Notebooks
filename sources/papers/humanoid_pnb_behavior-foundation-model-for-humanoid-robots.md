# Behavior Foundation Model for Humanoid Robots

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Behavior Foundation Model for Humanoid Robots
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Behavior_Foundation_Model_for_Humanoid_Robots/Behavior_Foundation_Model_for_Humanoid_Robots.html>
- **分类：** 03_High_Impact_Selection
- **子分类：** 仿真到现实与基座模型
- **arXiv：** <https://arxiv.org/abs/2509.13780>
- **入库日期：** 2026-06-07
- **一句话说明：** 把各类 WBC 任务都看成「在合适目标下生成行为轨迹」，先用 AMASS 重定向 + 仿真里特权信息的 proxy 运动模仿策略在线产出大规模行为数据，再用 掩码在线蒸馏 + 条件 VAE（CVAE） 学到可跨速度指令、遥操作、参考动作等多种控制接口共享的生成式策略，并可用 残差学习在不大改网络的前提下快速学会新动作——在仿真与真机上都展示了对多种全身任务的泛化与可组合潜空间。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-behavior-foundation-model-for-humanoid-robots](../../wiki/entities/paper-notebook-behavior-foundation-model-for-humanoid-robots.md).

## 对 wiki 的映射

- [paper-notebook-behavior-foundation-model-for-humanoid-robots](../../wiki/entities/paper-notebook-behavior-foundation-model-for-humanoid-robots.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Behavior_Foundation_Model_for_Humanoid_Robots/Behavior_Foundation_Model_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2509.13780>
