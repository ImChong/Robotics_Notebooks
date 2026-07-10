---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2507.11498"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robot-drummer.md
summary: "人形在灵巧、平衡、行走上进步显著，但在音乐表演等表现性领域的角色仍少被探索。本文提出 Robot Drummer，通过一连串定时接触完成打鼓，把问题表述成节奏接触链（Rhythmic Contact Chain）。系统把乐曲分解成定长片段，并行用强化学习训练。在 30+ 首摇滚、金属、爵士曲目上测试，取得高 F1 分数，并涌现出交叉臂击打（cross-arm strikes）与自适应鼓棒分配（adaptive stick assignments）等行为，能完成数分钟级的多肢协调演奏。"
---

# Robot Drummer

**Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形在灵巧、平衡、行走上进步显著，但在音乐表演等表现性领域的角色仍少被探索。本文提出 Robot Drummer，通过一连串定时接触完成打鼓，把问题表述成节奏接触链（Rhythmic Contact Chain）。系统把乐曲分解成定长片段，并行用强化学习训练。在 30+ 首摇滚、金属、爵士曲目上测试，取得高 F1 分数，并涌现出交叉臂击打（cross-arm strikes）与自适应鼓棒分配（adaptive stick assignments）等行为，能完成数分钟级的多肢协调演奏。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Rhythmic Contact Chain | 节奏接触链，一连串定时接触 |
| Timed Contact | 定时接触，在准确时刻击鼓 |
| F1 Score | 衡量击打准确性的指标 |
| Cross-Arm Strike | 交叉臂击打（涌现策略） |
| Stick Assignment | 鼓棒分配（哪只手打哪面鼓） |
| Parallel RL | 并行强化学习 |

## 为什么重要

- **把表现性任务转成"定时接触序列"**是可学化的关键抽象；
- **分段并行 RL**对长时程节奏任务有效；
- **表现性领域（音乐）**是高动态多肢协调的新颖试金石，呼应羽毛球/足球等体育任务；
- 涌现的拟人策略显示 RL 能发现高效协调方式。

## 解决什么问题

人形**音乐表演（打鼓）**少被探索，难点： - 打鼓是**精确定时的多肢接触**序列； - 乐曲**长时程**，直接 RL 难； - 要**多肢协调**（双臂 + 鼓棒分配）。

Robot Drummer 要：把打鼓建模成可学的**定时接触序列**，让人形演奏真实曲目。

## 核心机制

1. **节奏接触链表述**：把打鼓转成定时接触序列；
2. **分段并行 RL**：破解长时程乐曲；
3. **涌现拟人策略**：交叉臂击打、自适应鼓棒分配；
4. **真实曲目验证**：30+ 摇滚/金属/爵士高 F1。

方法拆解（深读笔记小节）：节奏接触链（Rhythmic Contact Chain）；分段并行 RL；涌现拟人策略；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming.html> |
| arXiv | <https://arxiv.org/abs/2507.11498> |
| 作者 | Asad Ali Shahid、Francesco Braghin、Loris Roveda（米兰理工 / IDSIA） |
| 发表 | 2025 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robot-drummer.md](../../sources/papers/humanoid_pnb_robot-drummer.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming.html>
- 论文：<https://arxiv.org/abs/2507.11498>

## 推荐继续阅读

- [机器人论文阅读笔记：Robot Drummer](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming.html)
