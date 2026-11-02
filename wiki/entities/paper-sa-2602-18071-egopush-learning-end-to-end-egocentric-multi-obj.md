---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2602.18071"
venue: "arXiv 2026"
summary: "A policy learning framework enabling egocentric, perception-driven rearrangement without relying on explicit global state estimation. Uses object-centric latent space to encode relative spatial relations"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2602_18071_egopush-learning-end-to-end-egocentric-m.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoPush

**EgoPush: Learning End-to-End Egocentric Multi-Object Rearrangement for Mobile Robots** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 082/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A policy learning framework enabling egocentric, perception-driven rearrangement without relying on explicit global state estimation. Uses object-centric latent space to encode relative spatial relations

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A policy learning framework enabling egocentric, perception-driven rearrangement without relying on explicit global state estimation. Uses object-centric latent space to encode relative spatial relations
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph P1["阶段 1：Teacher（仅训练，PPO 在线强化学习）"]
        A["仿真器稀疏关键点<br/>active / anchor / obstacle / target ref"] --> B["虚拟 FOV 遮挡 + 中心门控 Center-gated Visibility"]
        B --> C["共享权重 PointNet 编码各角色组"]
        C --> D["MLP 策略<br/>输入各组 latent + 上一步动作"]
        D --> E["动作 v, ω → PD 控制器 → 轮速"]
        R["分阶段奖励：reach / place 完成、进度 shaping、平滑、靠近目标减速"] --> D
    end
    subgraph P2["阶段 2：Student（在线 DAgger 蒸馏）"]
        F["自我中心 RGB-D"] --> G["实例分割 + 角色分组<br/>RGB 仅用于掩码"]
        G --> H["active / anchor / obstacle 三层 masked depth"]
        H --> I["CNN 编码各层"]
        I --> J["复制自 Teacher 的 MLP 主干 + 动作头"]
        K["自维护阶段计数器<br/>连续零动作帧推进并重路由角色掩码"] --> G
    end
    D -- "动作标签" --> L["蒸馏损失<br/>动作回归 + 关系余弦相似度矩阵对齐"]
    J --> L
    D -. "权重初始化（去掉 ref 列）" .-> J
    J --> M["部署：仅自我中心深度，零样本 sim-to-real"]
```

图依据：论文 §III-B/C/D / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 082/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.18071> |

## 核心机制（归纳）

### 策展导读要点

A policy learning framework enabling egocentric, perception-driven rearrangement without relying on explicit global state estimation. Uses object-centric latent space to encode relative spatial relations

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.18071) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoPush」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2602_18071_egopush-learning-end-to-end-egocentric-m.md`](../../sources/papers/sun_awesome_ego_2602_18071_egopush-learning-end-to-end-egocentric-m.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2602.18071>
- [arXiv PDF](https://arxiv.org/pdf/2602.18071)：已通读方法部分（实验与附录以略读为准）：§III Method（A–D）、Fig.2、Fig.3；§IV 实验仅略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2602.18071)
