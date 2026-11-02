---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-bridgeact-bridging-human-demonstrations-to-robot-actions
arxiv: "2604.23249"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_bridgeact-bridging-human-demonstrations-to-robot-actions.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# BridgeACT

**BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    subgraph D["(a) 训练数据自动构建（人类视频，无机器人数据）"]
        V1["VLM 任务理解<br/>抽取 tool/target 与动作类别"] --> V2["2D mask：已有标注或 SAM3<br/>CoTracker3 跟踪采样点"] --> V3["提升为 3D 轨迹并清洗<br/>样本 = 场景点云 + Q_tool + Q_target + 指令 l"]
    end
    subgraph GA["(b) Task-conditioned Affordance Grounding Agent"]
        G1["MLLM 规范化任务<br/>解析物体/区域描述与分割 prompt"] --> G2["SAM3 分割<br/>得到 M_tool 与 M_target"] --> G3["视觉语言 verifier 检查功能有效性"]
        G3 -->|不准确：至多一次回退 prompt 重新分割| G2
        G3 --> G4["投影到场景点云<br/>得到 3D 查询区域并采样查询点"]
    end
    IN["当前观测 + 任务指令"] --> G1
    G4 --> EN
    subgraph EM["(c) Embodiment-agnostic Affordance Representation"]
        EN["场景点与查询点分别嵌入<br/>PointNeXt 编码器-解码器，取查询点特征 H_Q"] --> FU["与 CLIP 语言特征 z_l 晚期融合得到条件 C"]
        FU --> DF["Transformer 扩散模型（前缀 token 条件）<br/>预测每个查询点的相对 3D 位移序列"]
    end
    D -.->|训练| EM
    DF --> PF["3D point flow"]
    PF --> ACT["动作生成：抓取用现成抓取模块；<br/>运动用 ICP 式刚体配准（SVD）求 SE(3) 变换，闭环重估"]
```

图依据：论文 §III-A 数据构建 / §III-B Grounding Agent / §III-C 表征与扩散预测 / §III-D 动作生成 / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | BridgeACT |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2604.23249> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2604.23249)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2604.23249) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「BridgeACT」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_bridgeact-bridging-human-demonstrations-to-robot-actions.md`](../../sources/papers/affordance_awesome_bridgeact-bridging-human-demonstrations-to-robot-actions.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2604.23249>
- [arXiv PDF](https://arxiv.org/pdf/2604.23249)：已通读方法部分（实验与附录以略读为准）：§III-A–§III-D、Fig.2

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2604.23249)
