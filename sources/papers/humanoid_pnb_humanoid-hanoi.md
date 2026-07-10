# Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2602.13850>
- **入库日期：** 2026-07-10
- **一句话说明：** 研究一个技能化（skill-based）的人形搬箱重排框架：通过在任务层把可复用技能串接来支持长时程执行。架构上，所有技能都经由一个「共享、任务无关的全身控制器（shared, task-agnostic WBC）」执行——这为技能组合提供了一致的闭环接口，区别于「每个技能各配一个低层控制器」的非共享设计。作者发现：直接朴素复用同一个预训练 WBC 会在长时程上削弱鲁棒性，因为新技能及其组合会引入偏移的状态与指令分布。他们用一个简单的数据聚合（data aggregation）过程来解决：把闭环技能执行在域随机化下的 rollout 回灌到共享 WBC 的训练里。为评估方法，他们提出 Humanoid Hanoi ——一个汉诺塔式的长时程箱体重排基准，并在仿真与 Digit V3 人形机器人上给出结果，展示了完全自主的长时程重排，并量化了共享 WBC 相对非共享基线的收益。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-humanoid-hanoi-investigating-shared-whole-body-c](../../wiki/entities/paper-notebook-humanoid-hanoi-investigating-shared-whole-body-c.md).

## 对 wiki 的映射

- [paper-notebook-humanoid-hanoi-investigating-shared-whole-body-c](../../wiki/entities/paper-notebook-humanoid-hanoi-investigating-shared-whole-body-c.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement.html>
- 论文：<https://arxiv.org/abs/2602.13850>
