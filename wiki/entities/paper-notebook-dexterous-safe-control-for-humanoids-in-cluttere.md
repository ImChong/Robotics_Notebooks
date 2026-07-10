---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2502.02858"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dexterous-safe-control-for-humanoids-in-cluttere.md
summary: "在真实应用中确保人形安全且不牺牲性能至关重要。本文考虑灵巧安全（dexterous safety）问题，特点是肢体级（limb-level）几何约束，用于在杂乱环境中同时避免外部碰撞与自碰撞。为处理\"确保碰撞避免\"时产生的大量约束，提出投影安全集算法（Projected Safe Set Algorithm, p-SSA）；针对约束不可行（infeasibility）问题，以有原则的方式松弛冲突约束，最小化安全违例以保证可行的机器人控制。在仿真与 Unitree G1 真机上验证：p-SSA 能让人形在挑战性场景中稳健运行、最小违例，并能跨任务免调参泛化。"
---

# Dexterous Safe Control for Humanoids in Cluttered Environments via Projected Safe Set Algorithm

**Dexterous Safe Control for Humanoids in Cluttered Environments via Projected Safe Set Algorithm** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

在真实应用中确保人形安全且不牺牲性能至关重要。本文考虑灵巧安全（dexterous safety）问题，特点是肢体级（limb-level）几何约束，用于在杂乱环境中同时避免外部碰撞与自碰撞。为处理"确保碰撞避免"时产生的大量约束，提出投影安全集算法（Projected Safe Set Algorithm, p-SSA）；针对约束不可行（infeasibility）问题，以有原则的方式松弛冲突约束，最小化安全违例以保证可行的机器人控制。在仿真与 Unitree G1 真机上验证：p-SSA 能让人形在挑战性场景中稳健运行、最小违例，并能跨任务免调参泛化。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| p-SSA | Projected Safe Set Algorithm，投影安全集算法 |
| Dexterous Safety | 灵巧安全（肢体级几何约束） |
| Limb-level | 肢体级，按各肢体几何建约束 |
| Self-collision | 自碰撞（机体各部分相撞） |
| Infeasibility | 约束不可行（冲突） |
| Safety Violation | 安全违例 |

## 为什么重要

- **"约束太多会不可行"是安全控制的真问题**，有原则的松弛比硬失败更实用；
- **肢体级几何**对高自由度人形的自碰撞避免必不可少；
- **免调参跨任务**的安全层利于工程复用；
- 与学习类控制互补：安全控制作"护栏"，学习作"性能"。

## 解决什么问题

人形在**杂乱环境**操作要**安全**： - 需**肢体级**避**外部碰撞 + 自碰撞**； - 约束**数量巨大**且可能**互相冲突（不可行）**； - 安全不能太保守而**牺牲性能**。

论文要：一个能处理**大量、可能冲突**约束、**最小违例**且**保性能**的安全控制算法。

## 核心机制

1. **灵巧安全问题**：肢体级几何约束，避外部 + 自碰撞；
2. **p-SSA 算法**：投影安全集处理大量约束；
3. **有原则松弛冲突约束**：最小化违例、保证可行控制；
4. **真机验证 + 免调参泛化**：G1 杂乱场景稳健。

方法拆解（深读笔记小节）：灵巧安全：肢体级几何约束；p-SSA：投影安全集算法；有原则地松弛冲突约束；验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set.html> |
| arXiv | <https://arxiv.org/abs/2502.02858> |
| 作者 | Rui Chen、Yifan Sun、Changliu Liu（CMU） |
| 发表 | 2025 年 2 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dexterous-safe-control-for-humanoids-in-cluttere.md](../../sources/papers/humanoid_pnb_dexterous-safe-control-for-humanoids-in-cluttere.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set.html>
- 论文：<https://arxiv.org/abs/2502.02858>

## 推荐继续阅读

- [机器人论文阅读笔记：Dexterous Safe Control for Humanoids in Cluttered Environments via Projected Safe Set Algorithm](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set/Dexterous_Safe_Control_for_Humanoids_in_Cluttered_Environments_via_Projected_Safe_Set.html)
