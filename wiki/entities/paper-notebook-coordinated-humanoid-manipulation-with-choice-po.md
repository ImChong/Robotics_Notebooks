---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2512.25072"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_coordinated-humanoid-manipulation-with-choice-po.md
summary: "人形要在人类环境中干活，难点是头、手、腿的全身协调。本文把模块化遥操作接口与可扩展学习框架结合：遥操作设计把人形控制分解为直观子模块——手眼协调、抓取基元、臂末端跟踪、行走，从而高效采集高质量演示。在此之上提出 Choice Policy：一种生成多个候选动作并学会给候选打分的模仿学习方法——既推理快又能建模多模态行为。在两个真实任务（洗碗机装载、擦白板的全身移动操作）上，Choice Policy 显著优于扩散策略与标准行为克隆；并发现手眼协调对长时程任务的成功至关重要。"
---

# Coordinated Humanoid Manipulation with Choice Policies

**Coordinated Humanoid Manipulation with Choice Policies** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形要在人类环境中干活，难点是头、手、腿的全身协调。本文把模块化遥操作接口与可扩展学习框架结合：遥操作设计把人形控制分解为直观子模块——手眼协调、抓取基元、臂末端跟踪、行走，从而高效采集高质量演示。在此之上提出 Choice Policy：一种生成多个候选动作并学会给候选打分的模仿学习方法——既推理快又能建模多模态行为。在两个真实任务（洗碗机装载、擦白板的全身移动操作）上，Choice Policy 显著优于扩散策略与标准行为克隆；并发现手眼协调对长时程任务的成功至关重要。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Choice Policy | 生成多个候选动作并学习打分的策略 |
| Modular Teleop | 模块化遥操作，把控制分解为直观子模块 |
| Grasp Primitive | 抓取基元，可复用的抓取动作单元 |
| End-Effector Tracking | 末端执行器跟踪，臂末端位姿跟踪 |
| Hand-Eye Coordination | 手眼协调，视觉与手部动作的配合 |
| Multimodal Behavior | 多模态行为，同一情境下多种合理动作 |

## 为什么重要

- **「生成候选 + 打分」是 BC 与扩散之间的折中**：保住多模态又不牺牲速度，值得在实时操作里推广；
- **模块化遥操作提升数据质量**：把高自由度控制拆成直观子模块，是人形数据采集的实用工程经验；
- **手眼协调被实证为长时程关键**：提示全身操作策略应显式建模视觉-手部耦合；
- **与 HumanProcessing / 末端跟踪类工作呼应**：如 HiWET、Learning Humanoid End-Effector Control。

## 解决什么问题

人形**全身协调**（头/手/腿）既难采数据又难学策略： - **采集难**：整体遥操作一个高自由度人形，难得到高质量演示； - **建模难**：操作行为常是**多模态**的（多种合理做法），单一回归/扩散各有取舍（扩散慢、BC 易塌缩到均值）。

论文要：一套**好采数据**的遥操作接口 + 一个**又快又能建多模态**的策略。

## 核心机制

1. **模块化遥操作接口**：把人形控制拆成手眼/抓取/末端/行走子模块，高效采高质量演示；
2. **Choice Policy**：生成多候选 + 学打分，兼顾快推理与多模态建模；
3. **真机验证**：洗碗机装载、擦白板全身操作上显著超越扩散与 BC；
4. **关键发现**：手眼协调是长时程任务成功的要素。

方法拆解（深读笔记小节）：模块化遥操作接口；Choice Policy：生成候选 + 学习打分；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Coordinated_Humanoid_Manipulation_with_Choice_Policies/Coordinated_Humanoid_Manipulation_with_Choice_Policies.html> |
| arXiv | <https://arxiv.org/abs/2512.25072> |
| 作者 | Haozhi Qi、Yen-Jen Wang、Toru Lin、Brent Yi、Yi Ma、Koushil Sreenath、Jitendra Malik（UC Berkeley） |
| 发表 | 2025 年 12 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_coordinated-humanoid-manipulation-with-choice-po.md](../../sources/papers/humanoid_pnb_coordinated-humanoid-manipulation-with-choice-po.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Coordinated_Humanoid_Manipulation_with_Choice_Policies/Coordinated_Humanoid_Manipulation_with_Choice_Policies.html>
- 论文：<https://arxiv.org/abs/2512.25072>

## 推荐继续阅读

- [机器人论文阅读笔记：Coordinated Humanoid Manipulation with Choice Policies](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Coordinated_Humanoid_Manipulation_with_Choice_Policies/Coordinated_Humanoid_Manipulation_with_Choice_Policies.html)
