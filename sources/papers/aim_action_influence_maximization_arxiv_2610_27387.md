# Action Influence Maximization 论文来源归档

## 元数据

- **题目：** Training Controllable World Models via Action Influence Maximization（方法简称 AIM）
- **作者：** Samuel M. Bateman, Tenny Yin, Chongyi Zheng, Lei Huang, Brian Wang, Benjamin Eysenbach, Jaime Fernández Fisac, Dhruv Shah
- **单位：** 抓取页面未列出
- **预印本：** https://arxiv.org/abs/2610.27387（cs.RO，v1，2026-10-19）
- **项目页 / 代码：** 抓取页面未列出
- **入库日期：** 2026-10-22

## 论文要解决的问题

动作条件的视频世界模型常忽略输入动作，倾向于预测专家式的典型未来。作者认为根因是训练数据的动作影响（action influence）不足，并证明这一数据限制会约束任何在其上训练的模型所能表达的动作影响。

## 方法要点（据 arXiv 摘要）

- 提出 Action Influence Maximization（AIM），主动采集训练 episode，提高数据集的动作影响。
- 核心判断：仅靠增大模型容量无法解决，需要改变数据本身。

## 报告的结果（据 arXiv 摘要）

- 像素级操作任务中，AIM 比探索与内在动机基线更数据高效地提升末端执行器可控性。

## Robotics_Notebooks 映射

- 实体页：[Action Influence Maximization](../../wiki/entities/paper-aim-action-influence-maximization.md)

## 一手来源

- arXiv 摘要页（经 WebFetch 摘要）
