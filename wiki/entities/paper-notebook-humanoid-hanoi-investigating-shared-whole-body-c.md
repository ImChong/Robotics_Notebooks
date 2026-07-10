---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2602.13850"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoid-hanoi.md
summary: "研究一个技能化（skill-based）的人形搬箱重排框架：通过在任务层把可复用技能串接来支持长时程执行。架构上，所有技能都经由一个「共享、任务无关的全身控制器（shared, task-agnostic WBC）」执行——这为技能组合提供了一致的闭环接口，区别于「每个技能各配一个低层控制器」的非共享设计。作者发现：直接朴素复用同一个预训练 WBC 会在长时程上削弱鲁棒性，因为新技能及其组合会引入偏移的状态与指令分布。他们用一个简单的数据聚合（data aggregation）过程来解决：把闭环技能执行在域随机化下的 rollout 回灌到共享 WBC 的训练里。为评估方法，他们提出 Humanoid Hanoi ——一个汉诺塔式的长时程箱体重排基准，并在仿真与 Digit V3 人形机器人上给出结果，展示了完全自主的长时程重排，并量化了共享 WBC 相对非共享基线的收益。"
---

# Humanoid Hanoi

**Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

研究一个技能化（skill-based）的人形搬箱重排框架：通过在任务层把可复用技能串接来支持长时程执行。架构上，所有技能都经由一个「共享、任务无关的全身控制器（shared, task-agnostic WBC）」执行——这为技能组合提供了一致的闭环接口，区别于「每个技能各配一个低层控制器」的非共享设计。作者发现：直接朴素复用同一个预训练 WBC 会在长时程上削弱鲁棒性，因为新技能及其组合会引入偏移的状态与指令分布。他们用一个简单的数据聚合（data aggregation）过程来解决：把闭环技能执行在域随机化下的 rollout 回灌到共享 WBC 的训练里。为评估方法，他们提出 Humanoid Hanoi ——一个汉诺塔式的长时程箱体重排基准，并在仿真与 Digit V3 人形机器人上给出结果，展示了完全自主的长时程重排，并量化了共享 WBC 相对非共享基线的收益。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| WBC | Whole-Body Controller，全身控制器 |
| Shared / Task-Agnostic | 共享 / 任务无关，一个控制器服务所有技能 |
| Skill Composition | 技能组合，把可复用技能按任务串接 |
| Long-Horizon | 长时程，需多步连续执行的任务 |
| Data Aggregation | 数据聚合（DAgger 式），把执行分布上的新数据回灌再训 |
| Domain Randomization | 域随机化，训练时随机化物理/环境参数以增强鲁棒 |
| Distribution Shift | 分布偏移，部署分布与训练分布不一致 |

## 为什么重要

- **「一个 WBC 服务所有技能」是可扩展技能系统的关键接口设计**：统一闭环接口让任务层能像搭积木一样组合技能；
- **长时程的敌人是分布偏移**：单技能好用 ≠ 串起来好用，组合分布必须被显式覆盖；
- **DAgger 式回灌依旧好使**：在执行分布上补数据这一经典思路，对人形长时程组合同样有效，工程代价低；
- **基准化很重要**：Humanoid Hanoi 这类「可量化长时程」的任务，有助于公平比较共享/非共享与不同数据策略，呼应本仓 11 仿真与基准板块。

## 解决什么问题

人形机器人做**长时程箱体重排**（搬来搬去、堆叠）需要把多个技能**连续串接**执行。一个自然的设计是：**让所有技能共用一个全身控制器**，从而获得统一的闭环组合接口。但这带来一个核心问题：

- **朴素复用预训练 WBC 会在长时程上掉鲁棒性**：当新技能与它们的**组合**被串起来时，会产生**状态分布**与**指令分布**的**偏移**，使原本好用的 WBC 逐步失稳。

## 核心机制

1. **共享、任务无关 WBC 的系统性研究**：作为技能组合的一致闭环接口，对照非共享设计；
2. **指明朴素复用的失效机理**：长时程技能组合引入状态/指令分布偏移，侵蚀鲁棒性；
3. **简单有效的数据聚合修复**：把域随机化下的闭环执行 rollout 回灌训练（DAgger 式）；
4. **Humanoid Hanoi 基准**：汉诺塔式长时程重排任务，仿真 + Digit V3 实测，量化共享 WBC 收益。

方法拆解（深读笔记小节）：架构：共享、任务无关的 WBC 作为统一接口；诊断：组合诱发的分布偏移；解法：闭环 rollout 的数据聚合（DAgger 式）；基准与评测：Humanoid Hanoi；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement.html> |
| arXiv | <https://arxiv.org/abs/2602.13850> |
| 机构 | Oregon State University（俄勒冈州立大学，作者群） |
| 作者 | Minku Kim、Kuan-Chia Chen、Aayam Shrestha、Li Fuxin、Stefan Lee、Alan Fern |
| 发表 | 2026 年 2 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoid-hanoi.md](../../sources/papers/humanoid_pnb_humanoid-hanoi.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement.html>
- 论文：<https://arxiv.org/abs/2602.13850>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoid Hanoi](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement/Humanoid_Hanoi__Investigating_Shared_Whole-Body_Control_for_Skill-Based_Box_Rearrangement.html)
