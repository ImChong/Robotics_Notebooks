# Google Research Blog：Barkour — 四足机器人动物级敏捷评测（2023-05-26）

> 官方博客来源归档（ingest）

- **标题：** Barkour: Benchmarking animal-level agility with quadruped robots
- **类型：** blog / official
- **URL：** <https://research.google/blog/barkour-benchmarking-animal-level-agility-with-quadruped-robots/>
- **作者署名：** Ken Caluwaerts、Atil Iscen（Research Scientists, Google；文内说明作者现属 Google DeepMind）
- **日期：** 2023-05-26
- **入库日期：** 2026-05-18
- **一句话说明：** 面向非专业读者重申 **障碍课 + 0–1 敏捷分**、**专长教师 RL（并行仿真）→ Transformer 学生蒸馏** 与 **导航控制器** 的部署管线，并强调 **小型犬约 10 s vs 机器人约 20 s** 的量级直觉、**恢复策略** 对反复实验的工程意义。

## 与论文互补的要点（博客侧重）

- **评测叙事：** 把分数与 ** novice 犬敏捷赛约 1.7 m/s** 目标速度对齐，强调 **可扩展障碍组合**。
- **系统切片：** 明确 **elevation map + 速度指令 + 机载传感** 作为部署时策略输入；**失败时恢复策略** + **走回起点** 减少人工干预。
- **实验观察：** 报告 **通才略低于专长切换平均分**，但 **行为/步态过渡更平滑**。

## 对 wiki 的映射

- [`wiki/entities/paper-barkour-quadruped-agility-benchmark.md`](../../wiki/entities/paper-barkour-quadruped-agility-benchmark.md)

## 当前提炼状态

- [x] 博客级要点与论文交叉索引
