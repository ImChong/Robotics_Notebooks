# Solving a Giant Rubik's Cube with a Quadruped Robot — 论文来源归档

> 来源归档（ingest）

## 元数据

- **题目：** Solving a Giant Rubik's Cube with a Quadruped Robot
- **作者：** Lukas Amiet、Gabriel Sánchez-Blanco Repgen、Mayank Mittal、Victor Klemm、Andrei Cramariuc、Marco Hutter、Cesar Cadena
- **机构：** ETH Zürich、NVIDIA
- **预印本：** <https://arxiv.org/abs/2610.24610>
- **项目页：** <https://leggedrobotics.github.io/rubik-skills/>
- **版本：** arXiv v1，2026-10
- **入库日期：** 2026-10-20
- **一句话说明：** 将机载 RGB-D 魔方感知、求解规划与强化学习足端操作技能串成四足机器人长时程自主任务。

## 核心论文摘录

### 问题与方法

四足机器人以足端操纵独立放置的多层魔方，必须在维持平衡的同时控制整体姿态与层状态；一次误转可能使后续解法失效。系统用 ANYmal-D 的 RGB-D 视角扫描魔方，估计几何位姿和颜色状态并融合多视角得到完整魔方状态；求解器输出解后，规划器将其转成移动、翻转和转层技能。技能包括前翻、后翻、逆时针转层，顺时针技能由逆时针策略镜像得到。策略在带特权观测的仿真中经强化学习训练，再蒸馏为以本体状态和深度图为输入的循环感知策略。

### 评测与结果

- 项目页报告几何位姿估计在 2 m 范围内的中位位置误差约 2 cm、旋转误差低于 3°；色块分类在室内序列为 100%，室外为 98.7%。
- 实机完成 3 次自主解魔方，每次约 30 个操作步骤、耗时 8–9 分钟。
- 消融实验指出，机载深度信息显著提升技能表现；循环记忆提高前翻与转层可靠性。

## 对 wiki 的映射

- [Rubik Skills 实体页](../../wiki/entities/paper-rubik-skills.md)
- [官方项目页归档](../sites/rubik-skills.md)

## 一手来源

- [论文（arXiv:2610.24610）](https://arxiv.org/abs/2610.24610)
- [项目页](https://leggedrobotics.github.io/rubik-skills/)
- [演示视频](https://www.youtube.com/watch?v=HFdRsFiE68c)
