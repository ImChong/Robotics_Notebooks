# FCTS（官方项目页）

> 来源归档（ingest）

- **标题：** FCTS: Failure-Guided Curriculum Tree Search Over Generated Terrain for Legged Parkour
- **类型：** site / project-page
- **URL：** <https://narayananpp.github.io/fcts/>
- **论文：** <https://arxiv.org/abs/2610.22619> — [论文摘录](../papers/fcts_failure_guided_curriculum_tree_search_arxiv_2610_22619.md)
- **代码 / 数据：** 截至 2026-10-20，项目页未列代码仓库或数据集链接。
- **入库日期：** 2026-10-20

## 一句话摘要

项目页以交互式流程和视频介绍 FCTS：验证失败驱动新地形生成，多个课程分支并行训练，再由树搜索选择表现更好的策略分支。

## 公开信息核查（2026-10-20）

- **源码状态：未发现公开入口。** 项目页可访问论文、视频和方法可视化，但页面链接中未列 GitHub 等代码仓库；论文摘要也未提供代码链接。
- **硬件信息：** 项目页称选定策略在 Jetson AGX Orin 上以 50 Hz 控制四足机器人，使用深度与本体感知，部署时未再调参。
- **结果口径：** 扫描场地的 held-out 路线成功率和真机演示属于不同验证；medium / hard 数值应按各自 course 与 baseline 理解。

## 关联资料

- 论文：[FCTS 论文摘录](../papers/fcts_failure_guided_curriculum_tree_search_arxiv_2610_22619.md)
- wiki：[FCTS 论文实体页](../../wiki/entities/paper-fcts-failure-guided-curriculum.md)
