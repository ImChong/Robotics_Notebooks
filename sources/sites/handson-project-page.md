# HandsOn 项目页与开源状态核查

> 来源归档

- **类型：** site（项目页入口）
- **URL：** https://llltttwww.github.io/HandsOn-website/
- **论文：** https://arxiv.org/abs/2610.26765
- **代码：** https://github.com/TsinghuaC3I/HandsOn
- **数据：** https://huggingface.co/datasets/Cyan666/HandsOn-Datasets
- **核查日期：** 2026-10-25

## 项目页核查

项目页将 HandsOn 描述为面向跨物体、环境与技能的统一灵巧操作仿真平台，列出 66 个任务、Core-50 与 Transfer-16 两个任务套件、18,000 条示范及评测结果；页面直接链接论文、代码与数据集。

项目页与 README 显示的开放边界：

- **任务与评测代码：已开源**，仓库包含仿真环境、任务定义、成功条件、统一评测入口及多种模型适配器。
- **示范数据：已发布 PhysX 版**，以 LeRobot v2.1 格式托管在 Hugging Face；Newton 后端数据仍待发布。
- **模型权重：未随仓库分发**，复现训练策略分数需要另行取得对应 checkpoint。
- **真机证据：有限**，项目展示定性 rollout；主要 benchmark 指标来自仿真闭环评测。

## Robotics_Notebooks 映射

- 论文：[handson_arxiv_2610_26765](../papers/handson_arxiv_2610_26765.md)
- 仓库：[handson](../repos/handson.md)
- 数据集：[handson_datasets](../datasets/handson_datasets.md)
- wiki 实体：[paper-handson](../../wiki/entities/paper-handson.md)
