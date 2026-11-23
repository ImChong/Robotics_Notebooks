# Humanoid Parkour Learning

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记）

- **标题：** Humanoid Parkour Learning
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Parkour_Learning/Humanoid_Parkour_Learning.html>
- **分类：** 03_High_Impact_Selection
- **子分类：** 行走经典
- **arXiv：** <https://arxiv.org/abs/2406.10759>
- **项目页：** <https://humanoid4parkour.github.io/>（[本地归档](../sites/humanoid-parkour-learning.md)）
- **源码：** 截至 2026-07-28 未公开；作者的 `ZiwenZhuang/parkour` 是四足前作，不是本文实现
- **入库日期：** 2026-06-07
- **复核日期：** 2026-11-21
- **核对范围：** 2026-11-21 通读 `arXiv:2406.10759v2`（CoRL 2024，pdftotext 1628 行）全文：Abstract、§1–§6（10 类公园地形、scandots 特权教师 + GRU 状态估计、多进程 DAgger 48×64 深度学生蒸馏、Unitree H1 0.42 m 跳台 / 0.8 m 跨沟 / 1.8 m/s 野外跑与上肢通道覆盖迁移、Table I–V 消融、Limitations）与 Appendix；对应人形官方训练/部署源码未公开。
- **通读：** 已完成
- **一句话说明：** 用 RL 在仿真里练「神谕」地形策略（scandots 地形编码 + GRU 状态估计），再通过 DAgger 把感知蒸馏成机载深度图 CNN，在 无参考轨迹、无抬脚奖励项 的前提下，让人形在多种跑酷障碍上 零样本 sim-to-real，并能跟随摇杆转向命令；手臂通道可覆盖以迁移到移动操作。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-humanoid-parkour-learning](../../wiki/entities/paper-notebook-humanoid-parkour-learning.md).

## 对 wiki 的映射

- [paper-notebook-humanoid-parkour-learning](../../wiki/entities/paper-notebook-humanoid-parkour-learning.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Parkour_Learning/Humanoid_Parkour_Learning.html>
- 论文：<https://arxiv.org/abs/2406.10759>
- 项目页：[humanoid-parkour-learning.md](../sites/humanoid-parkour-learning.md)
