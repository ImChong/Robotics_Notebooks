# EgoAlign 项目主页

- **标题：** EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation
- **类型：** site / project-page
- **URL：** <https://lambdahumanoid.github.io/EgoAlign/>
- **配套论文：** [arXiv:2609.38046](https://arxiv.org/abs/2609.38046) — [sources/papers/egoalign_arxiv_2609_38046.md](../papers/egoalign_arxiv_2609_38046.md)
- **代码仓：** <https://github.com/LambdaHumanoid/EgoAlign>（截至 2026-10-16，数据处理与仿真回放代码已公开；VLA 训练配方未发布）
- **作者 / 机构：** Yiming Jiang、Jin Chen、Chongyang Xu、Yilun Chen、Aimin Hao、Yisheng He；北京航空航天大学、上海创智学院、四川大学、阿里巴巴集团
- **入库日期：** 2026-10-03

## 一句话摘要

EgoAlign 通过双视角人体示范、目标机器人控制器反馈和因果状态重建，支持仅用适配后的人类示范微调 VLA，并在 G1 上零样本执行长程移动操作。

## 开源核查（2026-10-16）

项目页提供方法说明、演示视频和官方代码仓入口。官方仓 README 已包含 `endpoint-align`、`replay-only`、`sample_pipeline`、SONIC–MuJoCo 回放适配器及样例复现文档，可运行的数据处理与仿真回放代码已公开。VLA 训练方案仍列为未来可能发布；仓库未提供 EgoAlign 完整训练配方或权重，也未说明已发布完整示范训练集。**结论：部分开源，当前可复现数据处理和仿真回放，不能据此复现论文完整 VLA 训练。**

## 关联资料

- 论文归档：[sources/papers/egoalign_arxiv_2609_38046.md](../papers/egoalign_arxiv_2609_38046.md)
- 代码仓归档：[sources/repos/egoalign.md](../repos/egoalign.md)
- Wiki 实体页：[wiki/entities/paper-egoalign.md](../../wiki/entities/paper-egoalign.md)

