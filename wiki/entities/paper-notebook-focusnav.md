---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, unitree]
status: stub
updated: 2026-06-26
arxiv: "2601.12790"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_focusnav.md
summary: "FocusNav 把人形局部导航做成\"路径点先告诉我往哪走，注意力再去看那条路上的细节\"：用 WGSCA（路径点引导的空间交叉注意力）把感知聚焦到未来轨迹附近，用 SASG（稳定性感知选择门控）在打滑/失稳时主动屏蔽远端信息、把策略压回到脚下安全，在 Unitree G1 上显著提升复杂场景下的导航成功率。"
---

# FocusNav

**FocusNav: Spatial Selective Attention with Waypoint Guidance for Humanoid Local Navigation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

FocusNav 把人形局部导航做成"路径点先告诉我往哪走，注意力再去看那条路上的细节"：用 WGSCA（路径点引导的空间交叉注意力）把感知聚焦到未来轨迹附近，用 SASG（稳定性感知选择门控）在打滑/失稳时主动屏蔽远端信息、把策略压回到脚下安全，在 Unitree G1 上显著提升复杂场景下的导航成功率。

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
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local.html> |
| arXiv | <https://arxiv.org/abs/2601.12790> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_focusnav.md](../../sources/papers/humanoid_pnb_focusnav.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local.html>
- 论文：<https://arxiv.org/abs/2601.12790>

## 推荐继续阅读

- [机器人论文阅读笔记：FocusNav](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local.html)
