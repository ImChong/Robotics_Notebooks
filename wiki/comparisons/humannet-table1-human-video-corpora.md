---
type: comparison
tags: [dataset, egocentric-video, exocentric-video, vla, embodied-pretraining, humannet]
status: complete
updated: 2026-05-17
summary: "围绕 HumanNet 论文 Table 1，把代表性人类视频/行为语料按视点、活动语义与「具身向可用性」分组，并指向各数据集的官方入口；用于 VLA/模仿学习侧的人类数据选型，而非替代各数据集的官方数据卡。"
related:
  - ../entities/humannet.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/egoscale.md
  - ../concepts/embodied-scaling-laws.md
sources:
  - ../../sources/papers/humannet_table1_benchmark_corpora.md
  - ../../sources/papers/humannet.md
---

# HumanNet Table 1：代表性人类视频语料与具身向关系

**HumanNet** 在与既有语料对比时，用一张表同时强调 **规模、视点、活动语义粒度** 以及论文中称为 **Embodied Use** 的定性列（与「能否直接支撑机器人学习接口」相关，但仍是作者视角下的归类）。本页把该表 **提炼为阅读框架**，完整转录与链接索引见下方参考来源中的原始资料文件。

## 为什么重要？

- **人类视频 ≠ 同一种监督**：同样是「小时数」，厨房第一人称、电影第三人称与成对机器人示教，对 **接触几何、可执行动作标签、跨本体迁移** 的含义完全不同；Table 1 的价值在于把常见基准 **放在同一套维度下扫一眼**。
- **视点决定缺失信息**：第三人称擅长场景与全身运动上下文，但手指–物体接触常被遮挡；第一人称利于手–物关系，却可能缺少全局导航上下文——**Ego-Exo4D**、**HumanNet** 等混合视点路线试图折中。
- **Embodied Use 是「地图」不是「评分」**：论文用 Limited / Indirect / Direct 区分语料与机器人学习链路的距离；**Direct 也不等于开箱即用**，仍取决于标注字段、时间对齐、许可与分布偏移。

## 如何读论文里的三档 Embodied Use？

下列解释与 [原始资料](../../sources/papers/humannet_table1_benchmark_corpora.md) 中的表格转录一致，用于读论文时的直觉对齐（**非**对各数据集能力的客观排名）：

| 档位 | 直觉含义（结合 Table 1 语境） | 典型例子（论文表中） |
|------|------------------------------|----------------------|
| **Limited** | 行为相关，但与「可执行机器人接口」距离较远或需大量桥接 | EPIC-KITCHENS-100（厨房语义丰富，但论文仍标为 Limited） |
| **Indirect** | 适合表征 / 检测 / 视频理解预训练；要接到策略需额外对齐或生成式中间层 | ActivityNet、Kinetics、Ego4D、HowTo100M、Ego-Exo4D 等 |
| **Direct** | 更贴近 **轨迹级、手体级或成对机示教**，常被工作流直接当作模仿或预训练的人类侧信号 | HOI4D、EgoDex、OpenEgo、EgoScale、EgoVerse、Human2Robot、HumanNet |

**常见误区**：把 **Indirect** 当成「没用」——大规模第三人称语料仍是 VLM 与高层语义的重要底座；机器人侧难点通常在 **如何把间接监督压成可执行动作分布**。

## 视点与活动语义：三组速记

1. **Ego 灵巧线**：EgoDex / OpenEgo / EgoScale 等把 **第一人称 + 手部/任务结构** 推到数据建设前台，与 VLA 里「人视频小时 ↔ 验证损失」缩放叙事常一起出现（另见 [EgoScale](../methods/egoscale.md)）。
2. **Exo 互联网视频线**：Kinetics、Something-Something、HowTo100M 等覆盖 **语义与多样性**，更偏 **预训练与表征**，与真机日志的组合方式需单独设计。
3. **混合与成对**：Ego-Exo4D 强调 **多视点技能活动**；Human2Robot 强调 **人–机成对观测**；HumanNet 强调 **百万小时级人中心视频 + 交互向标注管线**（见 [HumanNet](../entities/humannet.md)）。

## 官方入口去哪找？

各数据集 **项目页、论文与常用下载入口** 已整理在单一索引页，避免本页变成纯外链列表：

- 见 [HumanNet Table 1 语料链接与规模转录](../../sources/papers/humannet_table1_benchmark_corpora.md)。

## 与其他页面的关系

- **实体**：[HumanNet](../entities/humannet.md) 给出语料定义、管线抽象与局限。
- **方法**：[VLA](../methods/vla.md)、[Imitation Learning](../methods/imitation-learning.md) 讨论人类侧数据与真机数据的互补与不等价替换。
- **方法**：[EgoScale](../methods/egoscale.md) 提供「万小时级 ego 人视频 ↔ VLA」的实证参照轴。
- **概念**：[具身规模法则](../concepts/embodied-scaling-laws.md) 用于讨论 **人视频小时** 与 **机器人日志小时** 在指标上不可混用。

## 参考来源

- [HumanNet Table 1 语料链接与规模转录](../../sources/papers/humannet_table1_benchmark_corpora.md)
- [HumanNet 论文 ingest 归档](../../sources/papers/humannet.md)
- Deng et al., *HumanNet: Scaling Human-centric Video Learning to One Million Hours*（[arXiv:2605.06747](https://arxiv.org/abs/2605.06747)）

## 关联页面

- [HumanNet](../entities/humannet.md)
- [VLA](../methods/vla.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [EgoScale](../methods/egoscale.md)

## 推荐继续阅读

- HumanNet 项目主页：<https://dagroup-pku.github.io/HumanNet/>（数据卡、发布说明与引用格式以官方为准）
