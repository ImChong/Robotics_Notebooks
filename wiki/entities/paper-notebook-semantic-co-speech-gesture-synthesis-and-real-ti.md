---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2512.17183"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_semantic-co-speech-gesture-synthesis-and-real-ti.md
summary: "论文把\"机器人讲话的同时做出语义对齐的手势\"这件事拆成 语义检索 + 自回归生成 + 人到机重定向 + 全身跟踪 四段流水线：用 LLM 从语料库里检索与语义高度相关的人体手势片段、用 Motion-GPT 自回归补全长时间序列、用 General Motion Retargeting (GMR) 把人体动作迁到 Unitree G1 上，最后用强化学习训出的 MotionTracker 把这套带有语义的参考动作在真机上稳定、实时地跟出来。"
---

# Semantic Co-Speech Gesture Synthesis and Real-Time Control for Humanoid Robots

**Semantic Co-Speech Gesture Synthesis and Real-Time Control for Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

论文把"机器人讲话的同时做出语义对齐的手势"这件事拆成 语义检索 + 自回归生成 + 人到机重定向 + 全身跟踪 四段流水线：用 LLM 从语料库里检索与语义高度相关的人体手势片段、用 Motion-GPT 自回归补全长时间序列、用 General Motion Retargeting (GMR) 把人体动作迁到 Unitree G1 上，最后用强化学习训出的 MotionTracker 把这套带有语义的参考动作在真机上稳定、实时地跟出来。

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
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2512.17183> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_semantic-co-speech-gesture-synthesis-and-real-ti.md](../../sources/papers/humanoid_pnb_semantic-co-speech-gesture-synthesis-and-real-ti.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2512.17183>

## 推荐继续阅读

- [机器人论文阅读笔记：Semantic Co-Speech Gesture Synthesis and Real-Time Control for Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots.html)
