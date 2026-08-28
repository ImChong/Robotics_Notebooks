# Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + 项目页/代码核查）

- **标题：** Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input/Learning_Agile_Striker_Skills_for_Humanoid_Soccer_Robots_from_Noisy_Sensory_Input.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2512.06571>
- **项目页：** <https://humanoidsoccer.github.io>
- **代码：** <https://github.com/Daffan/humanoid-soccer>
- **机构：** UT Austin · Sony AI
- **入库日期：** 2026-07-10
- **再核日期：** 2026-07-28
- **一句话说明：** 四阶段教师–学生连续踢球；仿真 SR 79.5%；Booster T1 真机 66.7%；已开源。

## 核心摘录（策展，非全文）

- Stage 1–2：特权教师追球 + 定向踢（Booster Gym 观测扩展）。
- Stage 3：DAgger；噪声 = 速度相关 + 延迟/异步 + 遮挡丢帧。
- Stage 4：N-P3O 约束精修；消融显示相对 PPO / 无 adaptation 显著增益。
- 真机：ZED 2i + YOLOv8 球检测；腿惯导门位；AGX Orin；五球位总 SR 66.7%。
- **开源核查（2026-07-28）：** 官方仓 Daffan/humanoid-soccer（`run.py` 四阶段入口）。

## 对 wiki 的映射

- [paper-notebook-learning-agile-striker-skills-for-humanoid-socce](../../wiki/entities/paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md)
- [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：见上
- 项目页：[humanoidsoccer-agile-striker.md](../sites/humanoidsoccer-agile-striker.md)
- 仓：[humanoid-soccer-agile-striker.md](../repos/humanoid-soccer-agile-striker.md)
- 论文：<https://arxiv.org/abs/2512.06571>
