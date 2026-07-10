---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.10635"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robot-crash-course.md
summary: "尽管行走越来越鲁棒，双足机器人在真实世界仍有跌倒风险。多数研究聚焦防摔，本文反其道专注「跌倒」本身：在减少机器人物理损伤的同时，给用户对机器人终止姿态（end pose）的控制权。为此提出一个机器人无关（robot-agnostic）的奖励函数，在 RL 中平衡三件事：达到期望终止姿态、冲击最小化、保护关键部件。为让策略对广泛的初始跌倒条件鲁棒、并能在推理时指定任意（甚至未见过）的终止姿态，引入一个基于仿真的初始/终止姿态采样策略。仿真与真机实验证明：双足机器人也能做受控的柔和跌倒。"
---

# Robot Crash Course

**Robot Crash Course: Learning Soft and Stylized Falling** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

尽管行走越来越鲁棒，双足机器人在真实世界仍有跌倒风险。多数研究聚焦防摔，本文反其道专注「跌倒」本身：在减少机器人物理损伤的同时，给用户对机器人终止姿态（end pose）的控制权。为此提出一个机器人无关（robot-agnostic）的奖励函数，在 RL 中平衡三件事：达到期望终止姿态、冲击最小化、保护关键部件。为让策略对广泛的初始跌倒条件鲁棒、并能在推理时指定任意（甚至未见过）的终止姿态，引入一个基于仿真的初始/终止姿态采样策略。仿真与真机实验证明：双足机器人也能做受控的柔和跌倒。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Soft Falling | 柔和跌倒，减小冲击的受控跌倒 |
| Stylized / End Pose | 可指定的终止姿态/落地造型 |
| Robot-Agnostic | 机器人无关，奖励不绑定特定本体 |
| Impact Minimization | 冲击最小化，降低落地受力 |
| Pose Sampling | 姿态采样，采样初始/终止姿态以泛化 |

## 为什么重要

- **「可指定终姿」拓展了跌倒安全的维度**：不仅少损伤，还能朝安全方向/造型摔；
- **机器人无关奖励**利于跨本体复用；
- **姿态采样**是获得鲁棒性与泛化（未见终姿）的简单有效手段；
- **与 SafeFall、自保护跌落、Unified Fall-Safety 同簇**，共同构成跌落安全研究群（本文出自 Disney/ETH 系，偏「风格化」控制）。

## 解决什么问题

防摔再好也无法**完全杜绝**跌倒。当跌倒发生时： - 要**减小损伤**； - 还希望能**指定落地终止姿态**（如朝某方向、保护某侧）； - 且要对**各种初始跌倒条件**都鲁棒、支持**任意未见终姿**。

论文要：让机器人学会**柔和且可指定姿态**的跌倒。

## 核心机制

1. **聚焦「跌倒本身」**：在防摔之外，研究怎么摔得柔和且可控；
2. **机器人无关三目标奖励**：终姿达成 + 冲击最小 + 护件；
3. **初始/终止姿态采样**：对广泛初始条件鲁棒、支持任意未见终姿；
4. **双足可受控软跌**：仿真与真机验证。

方法拆解（深读笔记小节）：机器人无关的三目标奖励；基于仿真的初始/终止姿态采样；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Robot_Crash_Course__Learning_Soft_and_Stylized_Falling/Robot_Crash_Course__Learning_Soft_and_Stylized_Falling.html> |
| arXiv | <https://arxiv.org/abs/2511.10635> |
| 作者 | Pascal Strauch、David Müller、Sammy Christen、Agon Serifi、Ruben Grandia、Espen Knoop、Moritz Bächer（Disney Research 等） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robot-crash-course.md](../../sources/papers/humanoid_pnb_robot-crash-course.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Robot_Crash_Course__Learning_Soft_and_Stylized_Falling/Robot_Crash_Course__Learning_Soft_and_Stylized_Falling.html>
- 论文：<https://arxiv.org/abs/2511.10635>

## 推荐继续阅读

- [机器人论文阅读笔记：Robot Crash Course](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Robot_Crash_Course__Learning_Soft_and_Stylized_Falling/Robot_Crash_Course__Learning_Soft_and_Stylized_Falling.html)
