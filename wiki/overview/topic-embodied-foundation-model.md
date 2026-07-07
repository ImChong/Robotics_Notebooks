---
type: overview
tags: [topic, embodied-foundation-model, vlm, vln, vla, vlx, world-model, taxonomy]
status: complete
updated: 2026-07-07
related:
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
  - ../concepts/3d-spatial-vqa.md
  - ../concepts/foundation-policy.md
  - ../concepts/world-action-models.md
  - ../concepts/latent-imagination.md
  - ../concepts/behavior-tree-vla-orchestration.md
  - ../concepts/humanoid-policy-network-architecture.md
  - ../concepts/visual-representation-for-policy.md
  - ../concepts/hierarchical-quadruped-navigation-stack.md
  - ../comparisons/humannet-table1-human-video-corpora.md
  - ./topic-vla.md
sources:
  - ../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md
summary: "具身大模型分类学选型闭环专题枢纽：把 VLM 感知理解 → VLN 空间导航 → VLA 动作执行 → VLX 一体化扩展 → 世界模型时序推演五大家族沉淀为一条贯通的选型链，统一各家族的 I/O 边界、数据需求、泛化能力与实时性取舍入口。"
---

# 具身大模型分类学选型闭环（专题汇总）

> **专题定位**：本页是「感知理解 → 空间导航 → 动作执行 → 一体化扩展 → 世界模型推演」五层具身大模型家族的统一入口，把近周密集 ingest 的 VLM / VLN / VLA / VLX / World-Model 五大家族从分散的实体/方法/对比页收拢为一条可导航的选型链。

## 一句话定义

**具身大模型分类学选型闭环** 指按 **跨模态感知（VLM）→ 空间导航（VLN）→ 动作执行（VLA）→ 一体化多任务扩展（VLX）→ 世界模型时序推演（WM）** 逐层分工的家族谱系，各层共享 Transformer 与多模态编码底座，但在 I/O 边界、数据需求、泛化能力与实时性上各有取舍，需按任务组合选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 跨模态感知理解骨干（图文对齐、空间 VQA） |
| VLN | Vision-Language Navigation | 语言指令驱动的空间导航 |
| VLA | Vision-Language-Action | 视觉-语言-动作统一策略，直接对接硬件 |
| VLX | Vision-Language-X | 一体化多任务扩展（X 为可插拔模态/技能头） |
| WM | World Model | 时序虚拟预演，与 VLA 形成决策–预演闭环 |
| WAM | World-Action Model | 把世界模型预测与动作生成合一的家族 |

## 为什么重要

- **补一条贯通的选型视角**：仓库已有各家族的实体/方法/对比页，但缺「从感知到推演逐层如何分工、各家族边界与取舍」的统一决策入口。
- **暴露家族间取舍矛盾**：端到端 VLA vs 分层 VLN、显式世界模型预测 vs 无模型反应式、大模型泛化 vs 实时控制带宽、统一 VLX vs 专精分立——这些矛盾只有并置在一条链上才看得清。
- **与人形产业叙事同向**：「大模型高层 + 实时低层控制」的分层部署，正是五层闭环在硬件上的落地形态。

## 五层选型闭环

| 层次 | 家族 | 典型职责 | 站内入口 |
|------|------|----------|----------|
| 感知理解 | VLM | 图文对齐、3D 空间 VQA | [3D 空间 VQA](../concepts/3d-spatial-vqa.md)、[策略视觉表征](../concepts/visual-representation-for-policy.md) |
| 空间导航 | VLN | 语言指令驱动导航 | [分层四足导航栈](../concepts/hierarchical-quadruped-navigation-stack.md) |
| 动作执行 | VLA | 视觉-语言-动作统一策略 | [VLA 专题](./topic-vla.md)、[基础策略](../concepts/foundation-policy.md)、[人形策略网络架构](../concepts/humanoid-policy-network-architecture.md) |
| 一体化扩展 | VLX | 多任务/多模态一体化 | [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md) |
| 世界模型推演 | WM / WAM | 时序预演与决策闭环 | [World Action Models](../concepts/world-action-models.md)、[潜空间想象](../concepts/latent-imagination.md) |
| 数据底座 | 跨层 | 人类视频语料等训练数据 | [HumanNet Table1 人类视频语料](../comparisons/humannet-table1-human-video-corpora.md) |

## 家族选型的关键取舍

- **端到端 vs 分层**：VLA 端到端泛化强但闭环稳定性与可解释性弱；VLN 分层栈稳定可控但泛化受限于模块接口。
- **显式世界模型 vs 无模型反应式**：WM/WAM 用预演换取长程规划能力，代价是推理延迟；VLA 反应式实时性好但缺长程预测。
- **大模型泛化 vs 实时控制带宽**：家族越靠上游（VLM/VLX）泛化越强、频率越低，越靠执行端（VLA + 低层控制器）越要压延迟。
- **统一 VLX vs 专精分立**：一体化模型省接口成本但单点能力上限受限；专精分立各自最优但需编排（如行为树）粘合。

## 与其他专题的关系

- **[VLA 与基础策略专题](./topic-vla.md)**：执行层的开源谱系与 BFM 身体接口。
- **[五大具身模型分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)**：本闭环的家族边界与递进关系原始对比。

## 关联页面

- [VLM/VLN/VLA/VLX/世界模型分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md)
- [基础策略（Foundation Policy）](../concepts/foundation-policy.md)
- [World Action Models](../concepts/world-action-models.md)
- [潜空间想象（Latent Imagination）](../concepts/latent-imagination.md)
- [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md)
- [人形策略网络架构](../concepts/humanoid-policy-network-architecture.md)
- [策略视觉表征](../concepts/visual-representation-for-policy.md)
- [分层四足导航栈](../concepts/hierarchical-quadruped-navigation-stack.md)
- [HumanNet Table1 人类视频语料](../comparisons/humannet-table1-human-video-corpora.md)

## 参考来源

- [wechat_shenlan_five_embodied_model_taxonomy.md](../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md) — 深蓝具身智能《五大具身模型详解：VLM、VLA、VLN、VLX、世界模型》
- 本页归纳自 [五大具身模型分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) 及各家族概念/方法页
