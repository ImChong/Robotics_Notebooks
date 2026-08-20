# grange007.github.io/HAF（HAF 项目页）

- **标题：** HAF — Humanoid Adaptation Framework
- **类型：** site / project-page
- **URL：** <https://grange007.github.io/HAF/>
- **入库日期：** 2026-08-20
- **配套论文：** [HAF（arXiv:2608.16837）](https://arxiv.org/abs/2608.16837) — 归档见 [`sources/papers/haf_arxiv_2608_16837.md`](../papers/haf_arxiv_2608_16837.md)
- **代码：** 截至入库日项目页 **未列出 GitHub / Hugging Face / 权重下载链接**（仅 arXiv 与演示视频）

## 一句话摘要

北大多媒体信息处理国重实验室等团队的 **HAF** 官方站点：展示 **HAF-VLA** 三阶段全身 action flow 与 **HAF-Steer** 频谱潜空间离线–在线 RL；七项天工人形家庭 loco-manipulation 真机任务对比 **π₀.₅ / GR00T N1.7 / Cosmos / ACT**。

## 公开信息要点（截至入库日）

- **核心叙事：** 把现成通才 flow-matching VLA 迁移到人形全身 loco-manipulation，无需大规模人形专属预训练；**HAF-VLA** 按运动学依赖分三阶段去噪（locomotion+head → +waist → +manipulation），**HAF-Steer** 在冻结生成器上用 flow reversal + DCT 压缩噪声子空间做 SAC 后训练。
- **演示板块：** Laundry Loading、Clothes Retrieval、Table Tidy、Basket Transfer、Toy Storage、Ball Tossing、Box Transfer 七任务真机视频；OOD 椅子干扰与起始位姿偏移；HAF-Steer 在 Toy Storage / Basket Transfer 上的离线–在线增益曲线。
- **资源链接：** arXiv（2608.16837）、PDF；**无 Code 按钮或仓库 URL**。

## 开源核查结论

| 项 | 状态 |
|----|------|
| 项目页 Code 区 | **未列出** |
| arXiv Code/Data 页 | **无官方仓库链接** |
| 结论 | **确认未开源**（截至 2026-08-20） |

## 为何值得保留

- **非 PDF 证据：** 七任务长程执行序列与 HAF-VLA / HAF-Steer 架构图比表格更直观呈现全身协调与后训练增益。
- **与论文三角互证：** 项目页主表平均成功率 **70.5%**（HAF-VLA）vs π₀.₅ **53.3%**、GR00T N1.7 **38.1%**，与 arXiv §5 一致。

## 关联资料

- 论文归档：[`sources/papers/haf_arxiv_2608_16837.md`](../papers/haf_arxiv_2608_16837.md)
- 机构：[`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)（天工 TienKung 2.0/3.0 平台）
