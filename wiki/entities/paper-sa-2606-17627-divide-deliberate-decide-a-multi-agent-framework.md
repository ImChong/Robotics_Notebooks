---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.17627"
venue: "arXiv 2026"
summary: "A fully-local, zero-shot multi-agent framework where VLM specialists engage in structured deliberation with peer consultation, improving zero-shot action recognition performance"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_17627_divide-deliberate-decide-a-multi-agent-f.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Divide, Deliberate, Decide

**Divide, Deliberate, Decide: A Multi-Agent Framework for Fine-Grained Egocentric Action Recognition** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 161/249** 篇，分组 **Action Recognition & Detection**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A fully-local, zero-shot multi-agent framework where VLM specialists engage in structured deliberation with peer consultation, improving zero-shot action recognition performance

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A fully-local, zero-shot multi-agent framework where VLM specialists engage in structured deliberation with peer consultation, improving zero-shot action recognition performance
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["第一人称视频 V + 动作标签集 Λ"] --> S1["Stage 1 Divide：切成最长 t_max 的 chunk<br/>降帧率并统一分辨率"]
    S1 --> ORC["VLM Orchestrator 把 chunk 划分为单动作 segment<br/>并给出每段初始 top-k 候选列表 Y(0)"]
    ORC --> PRI["Stage 2 Deliberate：各 VLM 专家独立给出<br/>自己的 top-k 先验排序"]
    PRI --> Q["先验共享；有分歧时<br/>每个专家至多提一个事实性问题，互相回答"]
    Q --> POST["所有专家看到完整交流后<br/>更新为后验排序（不强制一致）"]
    POST --> BORDA["Stage 3 Decide：Borda 计数聚合专家排序<br/>得到聚合 top-k"]
    BORDA --> FIN["Orchestrator 综合 Y(0)、各专家排序、聚合排序与 Borda 分数<br/>输出最终排序 Y(F)，不得引入新标签"]
    ORC --> FIN
```

图依据：论文 §3.1–3.3 / Fig.1（图本身未逐图核对，依正文）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 161/249 |
| 分组 | Action Recognition & Detection |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.17627> |

## 核心机制（归纳）

### 策展导读要点

A fully-local, zero-shot multi-agent framework where VLM specialists engage in structured deliberation with peer consultation, improving zero-shot action recognition performance

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.17627) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Action Recognition & Detection** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Divide, Deliberate, Decide」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action Recognition & Detection**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2606_17627_divide-deliberate-decide-a-multi-agent-f.md`](../../sources/papers/sun_awesome_ego_2606_17627_divide-deliberate-decide-a-multi-agent-f.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.17627>
- [arXiv PDF](https://arxiv.org/pdf/2606.17627)：已通读方法部分（实验与附录以略读为准）：§3 方法（Stage 1–3）、§4 实验设置

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.17627)
