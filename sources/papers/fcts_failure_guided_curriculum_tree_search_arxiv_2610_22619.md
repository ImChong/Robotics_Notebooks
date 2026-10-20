# FCTS: Failure-Guided Curriculum Tree Search Over Generated Terrain for Legged Parkour（官方论文）

> 来源归档（ingest）

- **标题：** FCTS: Failure-Guided Curriculum Tree Search Over Generated Terrain for Legged Parkour
- **类型：** paper / preprint
- **作者：** Narayanan Palghat Parameswaran、Dong-Ki Kim、Jay Patrikar、Santiago Gutierrez、Harshad Zade、Shayegan Omidshafiei、Harmish Khambhaita
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）、FieldAI
- **论文：** <https://arxiv.org/abs/2610.22619>
- **项目页：** <https://narayananpp.github.io/fcts/> — [项目页归档](../sites/fcts-project.md)
- **代码：** 截至 2026-10-20，项目页未列公开代码仓库。
- **入库日期：** 2026-10-20

## 核心内容

FCTS（Failure-Guided Curriculum Tree Search）用策略在验证地形上的失败位置、终止类型、机器人状态和周边几何条件化生成后续训练地形，并以蒙特卡洛树搜索在固定训练预算内探索多条课程改进轨迹。搜索树节点是策略 checkpoint，边代表完整训练轮次；项目页说明使用 PUCT 平衡探索与利用，并对节点价值做最大值回传。

论文在由真实施工场地扫描几何重建、且未用于课程生成和训练的路线评估中，报告 medium course 成功率由最强 parkour / LLM curriculum baseline 的 28.4% 提升至 85.8%，hard course 由 14.9% 提升至 44.7%。项目页称选出的策略在四足真机上以 50 Hz 运行，输入深度与本体感知；这属于硬件迁移验证，不应与 held-out 路线成功率混为一谈。

## 对 wiki 的映射

- [FCTS 论文实体页](../../wiki/entities/paper-fcts-failure-guided-curriculum.md) — 失败条件地形生成与预算约束课程搜索。
- [楼梯、障碍与感知运动任务页](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) — 与其他跑酷、复杂地形路线并列比较。
