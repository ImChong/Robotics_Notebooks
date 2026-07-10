---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2411.04005"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_object-centric-dexterous-manipulation-from-human.md
summary: "把物体操控到目标状态是灵巧操作的基本而重要的技能。人手动作展现了高超操控力，是训练多指手机器人的宝贵数据。本文通过分层策略弥合人手与机器人手的具身差距：① 高层——在大规模人手动捕数据上训练的轨迹生成模型，依目标物体状态合成手腕运动；② 低层——用深度强化学习做手指操控控制器，扎根于机器人本体。在 10 个家用物体上评测，对新物体几何与新目标状态泛化，并在双臂灵巧机器人系统上完成 sim-to-real。"
---

# Object-Centric Dexterous Manipulation from Human Motion Data

**Object-Centric Dexterous Manipulation from Human Motion Data** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

把物体操控到目标状态是灵巧操作的基本而重要的技能。人手动作展现了高超操控力，是训练多指手机器人的宝贵数据。本文通过分层策略弥合人手与机器人手的具身差距：① 高层——在大规模人手动捕数据上训练的轨迹生成模型，依目标物体状态合成手腕运动；② 低层——用深度强化学习做手指操控控制器，扎根于机器人本体。在 10 个家用物体上评测，对新物体几何与新目标状态泛化，并在双臂灵巧机器人系统上完成 sim-to-real。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Object-Centric | 以物体（目标状态）为中心 |
| Hierarchical | 分层（高层轨迹 + 低层手指） |
| Wrist Motion | 手腕运动（高层合成） |
| Finger RL | 手指控制的深度强化学习 |
| Embodiment Gap | 人-机手具身差距 |
| Goal State | 物体目标状态 |

## 为什么重要

- **"高层人类轨迹 + 低层机器人 RL"是弥合手部具身差距的经典分层**；
- **以目标状态为中心**让任务定义清晰、便于泛化；
- 人手动捕是灵巧操作的宝贵先验（与 EgoDex、Being-H0 同源思路）；
- 对人形双手灵巧操作直接适用。

## 解决什么问题

用人手动捕学机器人灵巧操作有**具身差距**： - 人手与机器人手**形态/自由度不同**； - 要**把物体操控到目标状态**，需手腕 + 手指协同； - 要对**新物体/新目标**泛化。

论文要：**分层**地用人手动捕学**以物体为中心**的灵巧操作。

## 核心机制

1. **以物体为中心的灵巧操作**：操控物体到目标状态；
2. **分层策略弥合具身差距**：高层人手动捕轨迹 + 低层手指 RL；
3. **泛化**：新物体几何与新目标状态；
4. **双臂 sim-to-real**：10 家用物体真机验证。

方法拆解（深读笔记小节）：高层：人手动捕轨迹生成（合成手腕运动）；低层：手指操控 RL（扎根机器人本体）；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Object-Centric_Dexterous_Manipulation_from_Human_Motion_Data/Object-Centric_Dexterous_Manipulation_from_Human_Motion_Data.html> |
| arXiv | <https://arxiv.org/abs/2411.04005> |
| 作者 | Yuanpei Chen、Chen Wang、Yaodong Yang、C. Karen Liu（Stanford / 北大） |
| 发表 | 2024 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_object-centric-dexterous-manipulation-from-human.md](../../sources/papers/humanoid_pnb_object-centric-dexterous-manipulation-from-human.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Object-Centric_Dexterous_Manipulation_from_Human_Motion_Data/Object-Centric_Dexterous_Manipulation_from_Human_Motion_Data.html>
- 论文：<https://arxiv.org/abs/2411.04005>

## 推荐继续阅读

- [机器人论文阅读笔记：Object-Centric Dexterous Manipulation from Human Motion Data](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Object-Centric_Dexterous_Manipulation_from_Human_Motion_Data/Object-Centric_Dexterous_Manipulation_from_Human_Motion_Data.html)
