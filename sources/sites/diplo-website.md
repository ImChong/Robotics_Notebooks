# DiPLO 官方项目页面（atarilab.github.io/diplo-website）

> 来源归档（ingest）

- **标题：** DiPLO — Self-distilled Diffusion Planning for Loco-manipulation
- **类型：** site / project-page
- **官网：** <https://atarilab.github.io/diplo-website/>
- **站点源码：** <https://github.com/Atarilab/diplo-website>
- **对应论文：** [arXiv:2610.24847v1](https://arxiv.org/html/2610.24847v1)
- **入库日期：** 2026-11-18
- **核对范围：** 阅读项目页 Method（Data、Pretraining、SD:DPO、DA:PPO）、Generalization、Real world、BibTeX 与资源链接；核对 Atarilab/diplo-website README 和根目录文件结构（静态网站代码）。论文细节以 arXiv v1 为准。
- **通读：** 已完成

## 一手资料摘要

项目展示 29 条人类运动经 IK/SBTO/对称性扩成 7,184 条、64 帧运动扩散规划、冻结 RL tracker 的 SD:DPO 自蒸馏及 DA:PPO 强化学习微调，并提供 G1 真机与未见目标、踢球、推拉物体的视频。

### 开放状态核查（2026-11-18）

- **论文：** arXiv v1 可公开阅读，尽管项目页导航仍显示 “Paper soon / arXiv soon”。
- **算法代码：** 项目页标记 “Code soon”；找到的 [Atarilab/diplo-website](https://github.com/Atarilab/diplo-website) 只有 README、index.html、src、static 等**展示站点源码**，并非 DiPLO 算法训练/推理仓库，不应误报已开源。
- **数据：** 项目页标记 “Dataset soon”。网站 README 提及 <https://huggingface.co/datasets/Atarilab/diplo-dataset>，但本次该地址无法核验到实际可下载条目；**只记录候选入口，不宣称数据可获得**。
- **模型/部署：** 未在官网看到可核验的 checkpoint 或 deploy scripts，因此不能依据视频重建运行时调用序列。

## 对 wiki 的映射

- [DiPLO 论文详情](../../wiki/entities/paper-diplo-self-distilled-diffusion-planning.md)
- [论文核对归档](../papers/diplo-self-distilled-diffusion-planning_arxiv_2610_24847.md)
