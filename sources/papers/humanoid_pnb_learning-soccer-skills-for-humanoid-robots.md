# Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + 项目页/代码核查）

- **标题：** Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr/Learning_Soccer_Skills_for_Humanoid_Robots____A_Progressive_Perception-Action_Fr.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2602.05310>
- **项目页：** <https://soccer-humanoid.github.io/>
- **代码：** <https://github.com/TeleHuman/HumanoidSoccer>
- **机构：** TeleAI（中国电信）· ShanghaiTech · ZJU · SJTU
- **入库日期：** 2026-06-07
- **再核日期：** 2026-07-28
- **一句话说明：** PAiD 三阶段：运动跟踪 → 感知融合 → 物理对齐；G1 仿真静球 SR 91.3%、滚动 71.9%；已开源。

## 核心摘录（策展，非全文）

- Stage I：13 条人类踢球 MoCap + GMR + 自适应采样统一跟踪（BeyondMimic 风格）。
- Stage II：骨盆系球/门观测 + 轻量任务奖励；LSTM 处理滚动球。
- Stage III：落球/滚动试验 + CMA-ES 接触参数对齐 + 物理引导观测噪声。
- Table IV：静球 91.3% / 0.9689；滚动 71.9% / 0.8892（有效工作区）。
- **开源核查（2026-07-28）：** 项目页 Code → TeleHuman/HumanoidSoccer。

## 对 wiki 的映射

- [paper-notebook-learning-soccer-skills-for-humanoid-robots](../../wiki/entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md)
- [paid-framework](../../wiki/methods/paid-framework.md)
- [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：见上
- 项目页归档：[soccer-humanoid-paid.md](../sites/soccer-humanoid-paid.md)
- 仓归档：[humanoid_soccer.md](../repos/humanoid_soccer.md)
- 论文：<https://arxiv.org/abs/2602.05310>
