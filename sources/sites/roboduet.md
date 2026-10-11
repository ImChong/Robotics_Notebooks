# RoboDuet 官方项目页来源归档

- **项目页：** https://locomanip-duet.github.io/
- **论文：** https://arxiv.org/abs/2403.17367
- **论文 DOI：** https://doi.org/10.1109/LRA.2025.3551230
- **训练代码：** https://github.com/locomanip-duet/RoboDuet
- **部署代码：** https://github.com/locomanip-duet/RoboDuet_Deployment
- **页面核查日期：** 2026-10-11

## 项目页核查摘要

项目页标注论文被 RA-L 2025 接收，并列出作者及机构（清华大学、上海期智研究院、上海人工智能实验室、江淮前沿技术中心、UC San Diego）。其方法说明为 locomotion policy 与 arm policy 协作，使用两阶段训练：先建立稳健盲行走能力，再学习移动操作的全身协调。

官方演示包括移动中跟踪末端 6D 位姿、坡面与碎石穿越、100 cm 高门把手开门、跨地面/椅面/柜面/站立桌面高度的物体搬运。页面报告相对 AFB+ID 基线提升 23% 成功率，以及 Go1+ARX5 训练策略直接迁移到 Go2+ARX5（底盘重量增加 14.7%）。

## 开源状态

**已开源（训练与部署实现）。** 项目页的 Code 链接指向训练仓库 `locomanip-duet/RoboDuet`；该仓库 README 再链接独立的 Go1/Go2 + ARX5 部署仓库 `RoboDuet_Deployment`。两个公开仓库均标注 MIT License。项目页未单列训练数据集或预训练模型下载链接；演示视频可作为定性结果，不等同于可下载数据集或权重。

## 对 wiki 的映射

- 项目实体：[RoboDuet](../../wiki/entities/paper-roboduet-loco-manipulation.md)
- 论文来源：[arXiv 论文归档](../papers/roboduet_arxiv_2403_17367.md)
- 训练仓库来源：[RoboDuet](../repos/roboduet.md)
- 部署仓库来源：[RoboDuet_Deployment](../repos/roboduet_deployment.md)
- 任务入口：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
