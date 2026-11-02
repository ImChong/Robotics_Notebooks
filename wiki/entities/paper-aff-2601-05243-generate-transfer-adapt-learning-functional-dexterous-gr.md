---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-generate-transfer-adapt-learning-functional-dexterous-gr
arxiv: "2601.05243"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: implicit；Action Subcategory: Hierarchical affordance-to-action · Retrieval-based）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_generate-transfer-adapt-learning-functional-dexterous-gr.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# CorDex

**Generate, Transfer, Adapt: Learning Functional Dexterous Grasping from a Single Human Demonstration**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generate, Transfer, Adapt: Learning Functional Dexterous Grasping from a Single Human Demonstration——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: implicit；Action Subcategory: Hierarchical affordance-to-action · Retrieval-based）**（清单未附方法摘要；方法流程见「流程总览」）。

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
    subgraph ENGINE["数据引擎（单段人类演示 → 合成训练数据）"]
        DEMO["单段人类演示视频"] --> GEN["生成多样物体：检索同类互联网图像<br/>（不足时 GPT-Image 补图）→ Rodin 2D-to-3D 生成网格"]
        DEMO --> CT["重建手网格与物体点云 → 提取 3D 指尖接触点"]
        GEN --> TR["2D 匹配（演示帧 ↔ 新物体多视角渲染）<br/>反投影到 3D，密度聚类取前 3 个簇中心为带置信度的候选接触点"]
        CT --> TR
        TR --> ADP["物理引导的抓取适配优化<br/>接触先验 / 稳定接触 / 辅助接触 / 关节限位 / 碰撞 / 自穿透损失"]
        ADP --> VER["仿真六方向外力验证稳定性"]
        VER --> RND["Blender 渲染 RGB-D 训练数据"]
    end
    RND --> TRAIN["训练 CorDex 抓取预测网络"]
    subgraph NET["CorDex 预测网络"]
        IN["单视角 RGB-D + 机器人手点云"] --> ENC["图像 / 点云编码器 → 逐点语义 + 几何特征"]
        ENC --> SAMP["重要性感知采样：4096 → 1024 点"]
        SAMP --> LG["局部-全局融合：自适应半径局部交叉注意力 + 全局自注意力"]
        LG --> XA["与机器人手特征交叉注意力 → 手-物距离矩阵"]
        XA --> OPT["多边测量 + 逆运动学恢复抓取 g"]
    end
    TRAIN -.-> NET
```

图依据：论文 §IV（基于对应关系的数据合成）/ Fig.2、§V（多模态融合抓取预测）/ Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | CorDex |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: implicit；Action Subcategory: Hierarchical affordance-to-action · Retrieval-based |
| 论文 | <https://arxiv.org/abs/2601.05243> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；机制见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2601.05243)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2601.05243) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「CorDex」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_generate-transfer-adapt-learning-functional-dexterous-gr.md`](../../sources/papers/affordance_awesome_generate-transfer-adapt-learning-functional-dexterous-gr.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2601.05243>
- [arXiv PDF](https://arxiv.org/pdf/2601.05243)：已通读方法部分（实验与附录以略读为准）：§IV-A/B/C 数据引擎、§V 多模态融合抓取预测网络

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2601.05243)
