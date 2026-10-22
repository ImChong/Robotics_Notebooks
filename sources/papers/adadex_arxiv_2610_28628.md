# AdaDex 论文来源归档

## 元数据

- **题目：** AdaDex: Learning Adaptable Dexterous Skills via Hybrid Dexterous Mimicking
- **作者与单位（项目页）：** Yuxuan Kuang（CMU、NVIDIA）、Jiashun Wang（CMU）、Yixuan Sun（CMU）、Yufei Ye（NVIDIA）、Sungjae Park（CMU）、Umar Iqbal（NVIDIA）、Katerina Fragkiadaki（CMU）、Shubham Tulsiani（CMU）
- **预印本：** https://arxiv.org/abs/2610.28628（cs.RO，v1，2026-10-20 提交）；PDF：https://arxiv.org/pdf/2610.28628
- **项目页：** https://yuxuank.com/AdaDex/
- **代码：** 项目页标注 "coming soon"，截至核查日无仓库；数据集链接未列
- **入库日期：** 2026-10-22

## 论文要解决的问题

灵巧手技能常需要大量示范或针对具体任务的设计；作者希望从单条数秒的手-物轨迹（动捕或视频重建）学到可泛化、可调整的灵巧技能。

## 方法要点（据摘要与项目页）

- 目标条件 RL 框架：输入单条手-物轨迹，学习可适应的灵巧技能。
- **Hybrid Dexterous Mimicking：** 一个共享策略在两类环境中训练：
  - 运动跟踪环境：注入人类运动先验；
  - 泛化环境：起点与目标加噪，并使用稀疏成功奖励。

## 报告的结果（据项目页）

- Seen 动作（GRAB，12 个）：进度 96.74%，ManipTrans 35.20%，SimToolReal 9.72%。
- 相似但未见动作（24 个）：进度 80.02%，ManipTrans 7.28%，SimToolReal 13.68%。
- 随机目标到达：进度 51.80%（SimToolReal 44.60%），**成功率 27.08%，低于 SimToolReal 的 31.71%**。
- 真实到仿真跟踪（9 段野外动作）：平均进度 80.53%，ManipTrans 31.70%。
- 迁移：Isaac Gym → MuJoCo 零样本闭环；真机 arm-hand 系统 30 Hz 执行。

## Robotics_Notebooks 映射

- 实体页：[AdaDex](../../wiki/entities/paper-adadex.md)

## 一手来源

- arXiv 摘要页与项目页（经 WebFetch 摘要）
