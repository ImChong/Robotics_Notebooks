---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2512.01336"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_discovering-self-protective-falling-policy-for-h.md
summary: "受形态、动力学与控制策略限制，人形比四足/轮式更容易摔；而其体重大、质心高、自由度高，不受控跌倒会对自身与周围造成严重硬件损伤。已有研究多用基于控制的方法，难以覆盖多样跌倒场景，且可能引入不合适的人类先验。本文转而用大规模深度强化学习 + 课程学习，激励人形自行探索贴合自身形态与属性的护身跌倒策略。通过精心设计的奖励与域多样化课程，成功训练智能体探索跌落保护行为，并发现：通过形成「三角（triangle）」结构，刚性机体的跌落损伤可被显著降低。论文用全面指标与实验量化其表现、可视化跌倒行为，并成功迁移到真实平台。"
---

# Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning

**Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

受形态、动力学与控制策略限制，人形比四足/轮式更容易摔；而其体重大、质心高、自由度高，不受控跌倒会对自身与周围造成严重硬件损伤。已有研究多用基于控制的方法，难以覆盖多样跌倒场景，且可能引入不合适的人类先验。本文转而用大规模深度强化学习 + 课程学习，激励人形自行探索贴合自身形态与属性的护身跌倒策略。通过精心设计的奖励与域多样化课程，成功训练智能体探索跌落保护行为，并发现：通过形成「三角（triangle）」结构，刚性机体的跌落损伤可被显著降低。论文用全面指标与实验量化其表现、可视化跌倒行为，并成功迁移到真实平台。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Self-Protective Falling | 自保护跌倒，受控地摔以减小损伤 |
| Deep RL | 深度强化学习 |
| Curriculum Learning | 课程学习，由易到难逐步训练 |
| Domain Diversification | 域多样化，多样化训练场景/参数 |
| CoM | Center of Mass，质心（人形质心高、易摔重） |
| Triangle Structure | 「三角」支撑结构，降低冲击的姿态 |

## 为什么重要

- **跌倒安全是人形落地的硬约束**：会摔不可避免，"摔得聪明"能省下大量硬件损耗；
- **让 RL 发现策略 > 人为规定姿态**：机器人物性与人不同，数据驱动的护身姿态更合适；
- **与 SafeFall、Robot Crash Course、VIGOR、Unified Fall-Safety Policy 同主题**，共同构成人形「跌落安全」研究簇；
- **域多样化课程**是覆盖长尾跌倒场景的实用手段。

## 解决什么问题

人形**易摔且摔得重**（体重大、质心高、自由度高），不受控跌倒会损坏硬件与环境。已有**控制法**： - **难覆盖多样跌倒**（方向/初速/地形各异）； - **易引入不当人类先验**（人为规定姿态未必适配机器人）。

论文要：让机器人**自己学**出**适配自身**的护身跌倒策略，最小化损伤并能上真机。

## 核心机制

1. **RL 自探索护身跌倒**：避开控制法的覆盖局限与不当人类先验；
2. **奖励 + 域多样化课程**：以损伤最小化为目标、覆盖多样跌倒；
3. **涌现「三角」结构**：贴合刚性机体物性、显著降损的策略发现；
4. **真机迁移**：全面量化并部署到真实平台。

方法拆解（深读笔记小节）：用 RL 探索而非人为规定；奖励设计 + 域多样化课程；涌现的「三角」结构；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Discovering_Self-Protective_Falling_Policy_for_Humanoid_Robot_via_Deep_RL/Discovering_Self-Protective_Falling_Policy_for_Humanoid_Robot_via_Deep_RL.html> |
| arXiv | <https://arxiv.org/abs/2512.01336> |
| 作者 | Diyuan Shi、Shangke Lyu、Donglin Wang（西湖大学） |
| 发表 | 2025 年 12 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_discovering-self-protective-falling-policy-for-h.md](../../sources/papers/humanoid_pnb_discovering-self-protective-falling-policy-for-h.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Discovering_Self-Protective_Falling_Policy_for_Humanoid_Robot_via_Deep_RL/Discovering_Self-Protective_Falling_Policy_for_Humanoid_Robot_via_Deep_RL.html>
- 论文：<https://arxiv.org/abs/2512.01336>

## 推荐继续阅读

- [机器人论文阅读笔记：Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Discovering_Self-Protective_Falling_Policy_for_Humanoid_Robot_via_Deep_RL/Discovering_Self-Protective_Falling_Policy_for_Humanoid_Robot_via_Deep_RL.html)
