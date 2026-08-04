---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2512.24657"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_antagonistic-bowden-cable-actuation-of-a-lightwe.md
summary: "用 拮抗式 Bowden 缆绳 + 滚动接触关节优化 把\"驱动电机\"全部搬到躯干，手部远端只剩 236 g 结构件却仍能输出 >18 N 指尖力、抓起 >100 倍自重 的负载 —— 给\"手臂载荷不够、手却必须像人手\"的人形机器人一条可工程化的路。"
---

# Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand

**Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

用 拮抗式 Bowden 缆绳 + 滚动接触关节优化 把"驱动电机"全部搬到躯干，手部远端只剩 236 g 结构件却仍能输出 >18 N 指尖力、抓起 >100 倍自重 的负载 —— 给"手臂载荷不够、手却必须像人手"的人形机器人一条可工程化的路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand.html> |
| arXiv | <https://arxiv.org/abs/2512.24657> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**这只手的取舍非常明确：用拮抗式 Bowden 缆绳把驱动电机整体外移到躯干，换取远端极低的质量——重量不是消失了，而是被转嫁到传动路径与躯干上。**

- 真正起作用的是两件事的组合：**拮抗式 Bowden 缆绳驱动 + 滚动接触关节优化**，使手部远端只剩 236 g 结构件，仍能输出 >18 N 指尖力、抓起 >100 倍自重的负载。
- 它瞄准的是一个具体的人形工程约束——「手臂载荷不够、手却必须像人手」：减掉的是远端质量与对手臂载荷的占用，而不是系统总重。
- 定位是 **硬件设计（12_Hardware_Design）** 而非学习方法，评价维度是机构学与驱动指标，与策略泛化类工作不可直接比较。
- 本页只是索引级实体：详细机制与完整量化指标、消融均未在此展开，不宜仅凭上述几个数字下工程可用性结论，需回到深读笔记与论文 PDF。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_antagonistic-bowden-cable-actuation-of-a-lightwe.md](../../sources/papers/humanoid_pnb_antagonistic-bowden-cable-actuation-of-a-lightwe.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand.html>
- 论文：<https://arxiv.org/abs/2512.24657>

## 推荐继续阅读

- [机器人论文阅读笔记：Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand.html)
