---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.03068"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_head.md
summary: "HEAD（Hand-Eye Autonomous Delivery）是一个直接从人类动作与视觉感知数据学人形导航、运动与触达的框架。采用模块化：高层规划器下达人形手与眼的目标位置与朝向，由低层策略控制全身动作来实现。具体地，低层全身控制器从大规模人类动捕学习跟踪三个点（双眼、左手、右手）；高层策略从 Aria 眼镜采集的人类第一视角数据学习。这种模块化把第一视角视觉感知与物理动作解耦，促进高效学习与对新场景的可扩展性。在仿真与真实世界评测，展示了人形在为人类设计的复杂环境中导航与触达的能力。"
---

# Hand-Eye Autonomous Delivery

**Hand-Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

HEAD（Hand-Eye Autonomous Delivery）是一个直接从人类动作与视觉感知数据学人形导航、运动与触达的框架。采用模块化：高层规划器下达人形手与眼的目标位置与朝向，由低层策略控制全身动作来实现。具体地，低层全身控制器从大规模人类动捕学习跟踪三个点（双眼、左手、右手）；高层策略从 Aria 眼镜采集的人类第一视角数据学习。这种模块化把第一视角视觉感知与物理动作解耦，促进高效学习与对新场景的可扩展性。在仿真与真实世界评测，展示了人形在为人类设计的复杂环境中导航与触达的能力。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| HEAD | Hand-Eye Autonomous Delivery |
| Reaching | 触达，手伸到目标位置 |
| Ego-centric | 第一视角（眼镜/头戴相机） |
| MoCap | 动作捕捉数据 |
| Three-Point Tracking | 三点跟踪：双眼 + 左右手 |
| Modular | 模块化，高/低层解耦 |

## 为什么重要

- **"眼+双手"三点是紧凑而强的全身目标表示**：抓住人类操作的关键端点，降低控制维度；
- **第一视角数据（Aria 眼镜）是高层策略的廉价来源**，呼应 ZeroWBC/EgoHumanoid 的第一视角路线；
- **感知-动作解耦**提升可扩展性，是模块化的经典收益；
- 把"导航 + 触达"统一在一个框架，贴近真实递送任务。

## 解决什么问题

人形要在**为人类设计的环境**里完成**递送**类任务，需要**导航 + 行走 + 触达**三种能力协同： - 端到端学习高自由度全身 + 感知很难、数据贵； - 想**复用海量人类数据**（动捕 + 第一视角）来学。

HEAD 要：用模块化把感知与动作解耦，**直接从人类数据**学出可导航、可触达的人形。

## 核心机制

1. **从人类数据学导航+运动+触达**：直接利用动捕与第一视角数据；
2. **模块化手眼框架**：高层下达手/眼目标、低层全身跟踪三点；
3. **感知-动作解耦**：提升学习效率与新场景可扩展性；
4. **仿真 + 真机**：人类环境中导航与触达验证。

方法拆解（深读笔记小节）：模块化：高层手眼目标 + 低层全身跟踪；低层：从动捕学三点跟踪；高层：从 Aria 第一视角学；解耦的好处 + 评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching.html> |
| arXiv | <https://arxiv.org/abs/2508.03068> |
| 作者 | Sirui Chen、Yufei Ye、Zi-Ang Cao、Jennifer Lew、Pei Xu、C. Karen Liu（Stanford） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_head.md](../../sources/papers/humanoid_pnb_head.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching.html>
- 论文：<https://arxiv.org/abs/2508.03068>

## 推荐继续阅读

- [机器人论文阅读笔记：Hand-Eye Autonomous Delivery](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching.html)
