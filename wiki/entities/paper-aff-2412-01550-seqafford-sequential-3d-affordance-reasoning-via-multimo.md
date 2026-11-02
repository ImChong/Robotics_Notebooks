---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-seqafford-sequential-3d-affordance-reasoning-via-multimo
arxiv: "2412.01550"
venue: "CVPR 2025"
summary: "CVPR 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: language-3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_seqafford-sequential-3d-affordance-reasoning-via-multimo.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# SeqAfford

**SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model**（CVPR 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: language-3D）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    PCL["目标物体点云 Xpoint"] --> MV["3D MLLM（ShapeLLM 主干）<br/>点云编码器 ReCon++ + 投影层"]
    INS["复杂人类指令 Xtxt"] --> LLM["大语言模型 LLaMA（LoRA 微调）"]
    MV --> LLM
    LLM --> RESP["文本回复：分步推理 + 多个 SEG token<br/>每个 SEG 对应一步 affordance 分割"]
    RESP --> HSEG["取各 SEG 的末层 embedding<br/>MLP Proj → Hseg"]
    PCL --> ENC["3D 视觉编码器<br/>中间层特征 + 稀疏全局特征 Fsparse"]
    ENC --> MGP["多粒度特征传播<br/>FPS + 上采样逐级传播 → 稠密特征 Fdense"]
    MGP --> PLI["点-语言融合<br/>Hseg 为 Q、Fdense 为 K/V 做 cross-attention + FFN<br/>再与 Fsparse 融合 → Af"]
    HSEG --> PLI
    ENC --> PLI
    PLI --> DEC["Decoder"]
    DEC --> MASK["按顺序的多个 affordance mask"]
    MASK --> LOSS["训练：文本自回归交叉熵 + 分割 BCE + Dice"]
```

图依据：论文 §4.1 架构概述、§4.2 / Fig.3（主流程）与 Fig.4（Multi-Granular Language-Point Integration）、§4.3 训练目标

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2025 |
| 方法名（清单标注） | SeqAfford |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: language-3D |
| 论文 | <https://arxiv.org/abs/2412.01550> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（仅画论文方法部分明确写出的模块与数据流，不含推测），训练细节与实验请读 [原文](https://arxiv.org/abs/2412.01550)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2412.01550) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「SeqAfford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_seqafford-sequential-3d-affordance-reasoning-via-multimo.md`](../../sources/papers/affordance_awesome_seqafford-sequential-3d-affordance-reasoning-via-multimo.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2412.01550>
- [arXiv PDF](https://arxiv.org/pdf/2412.01550)：已通读方法部分（实验与附录以略读为准）：§4（架构概述、网络结构、训练目标）与附录 E.1 点上采样开头；§3 数据集与 §5 实验略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2412.01550)
