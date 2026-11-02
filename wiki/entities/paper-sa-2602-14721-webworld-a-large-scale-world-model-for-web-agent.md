---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.14721"
venue: "arXiv 2026"
summary: "Built from over one million real web interactions; outperforms GPT-5 as a world model for inference-time search."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_14721_webworld-a-large-scale-world-model-for-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WebWorld

**WebWorld: A Large-Scale World Model for Web Agent Training** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 517/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Built from over one million real web interactions; outperforms GPT-5 as a world model for inference-time search.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Built from over one million real web interactions; outperforms GPT-5 as a world model for inference-time search.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["数据构建（分层采集）"]
        A["目标 URL：FineWeb / CCI 3.0 等预训练语料 + 高流量网站清单"] --> B["URL 过滤：可达性、关键词、LLM 四维评分"]
        B --> C1["Level 1 随机爬取<br/>从 A11y Tree 随机采样动作，3-10 步"]
        B --> C2["Level 2 LLM 智能体自主探索<br/>最长 30 步"]
        B --> C3["Level 3 任务导向执行<br/>种子任务 → 多样化 → 改写，只保留成功轨迹"]
        C1 --> D["轨迹规则过滤<br/>剔除无状态变化、超 30k token 或 30 轮"]
        C2 --> D
        C3 --> D
        D --> E["数据增强：多格式状态转换（HTML、XML、Markdown 等）、网页生成、描述式模拟、通用数据"]
    end
    subgraph TRAIN["两阶段训练"]
        F["阶段 1：全量数据最大似然训练<br/>P(s_t+1 | 指令 I, 历史 h_t)"] --> G["阶段 2：少量 CoT 数据继续训练<br/>激活推理"]
    end
    E --> F
    G --> W["WebWorld 世界模型（8B / 14B / 32B）"]
    subgraph USE["下游使用"]
        W --> H["Abstract-and-Instantiate 合成 8000 条轨迹<br/>→ 拒绝采样 → 微调 Qwen3-8B 智能体"]
        W --> I["推理期 lookahead 搜索<br/>候选动作 → WebWorld 模拟下一状态 → 价值模型打分 → 执行最高分动作"]
    end
```

图依据：论文 §3.1–3.5、§5.1–5.2 / Fig.1c

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 517/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.14721> |

## 核心机制（归纳）

### 策展导读要点

Built from over one million real web interactions; outperforms GPT-5 as a world model for inference-time search.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.14721) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WebWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **83 GUI / Computer-Use Agents**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2602_14721_webworld-a-large-scale-world-model-for-w.md`](../../sources/papers/sun_awesome_wm_2602_14721_webworld-a-large-scale-world-model-for-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.14721>
- [arXiv PDF](https://arxiv.org/pdf/2602.14721)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 相关工作、§3 Training WebWorld（3.1–3.6）、§5.1–5.2 的下游使用描述；其余实验与附录略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.14721)
