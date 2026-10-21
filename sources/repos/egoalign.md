# LambdaHumanoid/EgoAlign

- **标题：** EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation
- **类型：** repo / project-page-source / humanoid / loco-manipulation
- **URL：** <https://github.com/LambdaHumanoid/EgoAlign>
- **项目页：** <https://lambdahumanoid.github.io/EgoAlign/>（归档见 [sources/sites/egoalign-project.md](../sites/egoalign-project.md)）
- **论文：** [arXiv:2609.38046](https://arxiv.org/abs/2609.38046)（归档见 [sources/papers/egoalign_arxiv_2609_38046.md](../papers/egoalign_arxiv_2609_38046.md)）
- **机构：** 北京航空航天大学（Beihang University）；上海创智学院（Shanghai Innovation Institute）；四川大学（Sichuan University）；阿里巴巴集团（Alibaba Group）
- **入库日期：** 2026-10-03

## 一句话摘要

EgoAlign 官方实现仓，公开人体动作适配、因果回放与机器人状态—动作数据导出流程。

## 开源核查（2026-10-16）

仓库 README 和文档提供 Python CLI（`endpoint-align`、`replay-only`）、`sample_pipeline`、`scripts/human_data/`、SONIC–MuJoCo 回放运行适配器，以及处理、验证和样例复现指南。公开代码可用于数据构造与仿真回放；运行仍需按文档配置外部 SONIC/G1 资产、控制器权重及兼容环境。

README 将 VLA 训练配方列为未来可能发布，当前未提供 EgoAlign 的完整 VLA 训练代码或权重；仓库有示例任务文件，但未声明完整规模的人类示范训练集公开。**结论：部分开源，已提供可运行的数据处理和仿真回放代码，训练配方、权重与完整数据集可用性有限。**

## 关联资料

- 项目页：[sources/sites/egoalign-project.md](../sites/egoalign-project.md)
- 论文：[sources/papers/egoalign_arxiv_2609_38046.md](../papers/egoalign_arxiv_2609_38046.md)
- Wiki 实体页：[wiki/entities/paper-egoalign.md](../../wiki/entities/paper-egoalign.md)

