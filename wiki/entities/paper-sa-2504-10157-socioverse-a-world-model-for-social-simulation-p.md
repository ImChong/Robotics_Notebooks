---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2504.10157"
venue: "arXiv 2025"
code: https://github.com/FudanDISC/SocioVerse
summary: "World model for social simulation powered by LLM agents and 10 million real-world users."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2504_10157_socioverse-a-world-model-for-social-simu.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# SocioVerse

**SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 529/571** 篇，分组 **85 Social & Economic Simulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World model for social simulation powered by LLM agents and 10 million real-world users.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World model for social simulation powered by LLM agents and 10 million real-world users.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    SE["Social Environment<br/>社会结构、社会动态(带时间戳与标签的新闻)、个性化上下文"]
    UE["User Engine<br/>1000 万用户池(X、Rednote)+ 15 个人口统计维度标签<br/>LLM 初标 → 人工校验 → 训练分类器大规模标注"]
    SC["Scenario Engine<br/>问卷 1-to-N / 深度访谈 1-to-1<br/>行为实验 / 社交媒体交互 N-to-N<br/>按人口分布采样放大"]
    UE -->|"用户画像与历史经验"| BE
    SC -->|"模拟设定与交互机制"| BE
    SE -->|"最新社会信息与上下文"| BE
    BE["Behavior Engine<br/>Agent 模型池：传统 ABM(规则/数学模型)<br/>+ 通用 / 专家 / 领域 LLM"]
    BE --> OUT["对查询生成模拟行为结果"]
    OUT --> APP["三个应用场景：美国大选预测<br/>突发新闻反馈、中国国民经济调查"]
```

图依据：论文 §2 Methods（Overall Framework、2.1–2.4）/ Fig.2；§3 场景实现仅用于末节点

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 529/571 |
| 分组 | 85 Social & Economic Simulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2504.10157> |
| 代码/项目 | <https://github.com/FudanDISC/SocioVerse> |

## 核心机制（归纳）

### 策展导读要点

World model for social simulation powered by LLM agents and 10 million real-world users.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2504.10157) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **85 Social & Economic Simulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SocioVerse」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **85 Social & Economic Simulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2504_10157_socioverse-a-world-model-for-social-simu.md`](../../sources/papers/sun_awesome_wm_2504_10157_socioverse-a-world-model-for-social-simu.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2504.10157>
- [arXiv PDF](https://arxiv.org/pdf/2504.10157)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§2 Methods（2.1–2.4）、§3 开头

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2504.10157)
