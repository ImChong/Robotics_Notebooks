---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-uniafford-token-routed-multitask-learning-for-generaliza
arxiv: "2609.37264"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_uniafford-token-routed-multitask-learning-for-generaliza.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# UNIAFFORD

**UNIAFFORD: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

UNIAFFORD: Token-Routed Multitask Learning for Generalizable 2D-3D Affordance Perception——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要；方法流程见下方「流程总览」）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 流程总览

```mermaid
flowchart TD
    IN["输入：语言指令 + RGB 图像 和/或 点云<br/>(可缺失模态)"] --> ENC["模态编码<br/>文本分词 / SigLIP 图像编码 / SONATA 点云编码 + 投影"]
    ENC --> PRE["拼接前缀 T_in = 文本 + 图像 + 点云"]
    PRE --> LLM["MLLM 骨干（共享语义枢纽，LoRA）<br/>自回归生成响应隐状态 h_t"]
    LLM --> RT["模态感知 Token Router<br/>逐状态预测 text / img / pc，屏蔽不可用分支"]
    RT -->|text 状态| LH["语言头：文本输出"]
    RT -->|img 状态| QI["投影为 2D 亲和查询 Q_img"]
    RT -->|pc 状态| QP["投影为 3D 亲和查询 Q_pc"]
    QI --> D2["SAM 式 2D 解码器<br/>查询-特征相似度热图 → 掩码提示 → 像素级亲和掩码"]
    QP --> D3["SONATA 式 3D 解码器<br/>查询-点特征相似度 → 逐点亲和 logits"]
```

图依据：论文 §4.1 至 §4.4 与 Fig. 2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | UNIAFFORD |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2609.37264> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法模块与数据流见上方「流程总览」（依据论文方法部分）。训练细节与实验请读 [原文](https://arxiv.org/abs/2609.37264)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2609.37264) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「UNIAFFORD」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 清单可确证的只有标题、出处与分类标签；方法流程见「流程总览」，实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/affordance_awesome_uniafford-token-routed-multitask-learning-for-generaliza.md`](../../sources/papers/affordance_awesome_uniafford-token-routed-multitask-learning-for-generaliza.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2609.37264>
- [arXiv PDF](https://arxiv.org/pdf/2609.37264)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 数据集与任务定义、§4 Method（4.1–4.5）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2609.37264)
