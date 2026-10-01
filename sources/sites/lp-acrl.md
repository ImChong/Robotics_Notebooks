# LP-ACRL（Scaling Rough Terrain Locomotion）

> 来源归档

- **标题：** LP-ACRL: Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning
- **类型：** site（Google Sites 项目页）
- **论文：** https://arxiv.org/abs/2601.17428
- **PDF：** https://arxiv.org/pdf/2601.17428
- **DOI：** https://doi.org/10.1109/LRA.2026.3703486
- **项目页：** https://sites.google.com/view/lp-acrl
- **机构页：** https://rsl.ethz.ch/publications-sources/publications.html（RSL 出版物索引）
- **机构：** 苏黎世联邦理工学院机器人系统实验室（Robotic Systems Lab, ETH Zurich）；Chenhao Li 亦隶属 ETH AI Center
- **入库日期：** 2026-10-01
- **一句话说明：** 基于 episodic reward 学习进度（LP）在线重分配任务采样，在 600 实例多轴足式任务空间上自动课程 RL，ANYmal D 真机 rough terrain 高速 locomotion 部署（Teacher–Student 蒸馏学生策略）。

---

## 项目页要点（步骤 2.5 核查 · 2026-10-01）

- **Paper 区：** 链到 arXiv / RA-L DOI；含 Overview、Multi-Axis Difficulty、仿真与真机结果图与 **YouTube 演示**（项目页嵌入 `rr05THCb6Mg`）。
- **方法叙事：** 强调 **无需手工难度排序** 的非结构化任务空间；LP-ACRL 用学习进度 softmax 更新离散任务实例采样分布。
- **仿真栈（论文正文）：** Isaac Lab；训练框架与 Rudin et al. / Schwarke et al. 系 rough terrain locomotion 设定一致；底层 RL 实现生态上通常对接 [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)（**非** LP-ACRL 专用 fork）。

## 开源状态

- **核查范围：** 项目页 HTML、arXiv 摘要页、ETH RSL publications 索引、`leggedrobotics` GitHub 组织公开检索（无 `lp-acrl` / 2601.17428 专用仓）。
- **已发布：** 论文 PDF、项目页视频与图示、RA-L DOI。
- **未发布：** **无** 官方 LP-ACRL 训练/课程模块仓库；**无** 官方策略权重下载。
- **结论：** **确认未开源**（截至 2026-10-01）。复现需自研 Isaac Lab 任务离散化 + LP 采样逻辑；可参考 RSL 公开 **rsl_rl** 与 ANYmal 系 prior 工作管线。

## 对 wiki 的映射

- 论文实体：[paper-lp-acrl-scaling-rough-terrain-locomotion](../../wiki/entities/paper-lp-acrl-scaling-rough-terrain-locomotion.md)
- 源归档：[lp_acrl_arxiv_2601_17428.md](../papers/lp_acrl_arxiv_2601_17428.md)
- 概念交叉：[curriculum-learning](../../wiki/concepts/curriculum-learning.md)
