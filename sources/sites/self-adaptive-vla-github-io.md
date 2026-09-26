# icefoxzhx.github.io/self-adaptive-vla（Self-Adaptive VLA 项目页）

- **标题：** Self-Adaptive VLA for Robust Robot Deployment
- **类型：** site / project-page
- **URL：** <https://icefoxzhx.github.io/self-adaptive-vla/>
- **配套论文：** [Self-Adaptive VLA（arXiv:2609.30092）](https://arxiv.org/abs/2609.30092)
- **代码：** 页内 **无** GitHub / Hugging Face 链接（截至 2026-09-26）
- **入库日期：** 2026-09-26

## 一句话摘要

UMass + Genesis AI：**失败 rollout 作 context** + **shift 预补偿专家示范** + **AdaLN context token**；测试时 **ensemble 多次失败 token** 逐步恢复精密操纵；项目页以 **Assemble Ring / 新工位 / Piper 臂** 视频为主证据。

## 公开信息要点（截至入库日）

- **三块机制图：** contextualized data · context encoder · iterative ensembling。
- **Case study：** 20-DoF 灵巧手 joint encoder offset — Trial0 失败 → Trial1+context → Trial2 成功。
- **Station 迁移：** 仅 Station1 训练，Station2 base 全败，1–2 次失败 context 恢复。
- **BibTeX：** `@article{zhang2026selfadaptive,...}`。

## 关联资料

- 论文归档：[`sources/papers/self_adaptive_vla_arxiv_2609_30092.md`](../papers/self_adaptive_vla_arxiv_2609_30092.md)
