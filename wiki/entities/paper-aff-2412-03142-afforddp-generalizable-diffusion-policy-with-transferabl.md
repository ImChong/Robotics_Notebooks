---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-afforddp-generalizable-diffusion-policy-with-transferabl
arxiv: "2412.03142"
venue: "CVPR 2025"
summary: "CVPR 2025 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: explicit；Action Subcategory: Integrated policy · Explicit input）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_afforddp-generalizable-diffusion-policy-with-transferabl.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordDP

**AffordDP: Generalizable Diffusion Policy with Transferable Affordance**（CVPR 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordDP: Generalizable Diffusion Policy with Transferable Affordance——在清单中归入 **Action（Architecture: E2E；Affordance Usage: explicit；Action Subcategory: Integrated policy · Explicit input）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    subgraph MEM["Affordance memory M"]
        DEMO["演示：GroundedSAM 裁剪物体 + CLIP 图像特征 z<br/>记录接触点 c（静态）与接触后末端轨迹 τ（动态）<br/>及点云 P"]
    end
    TGT["目标物体图像 + 任务 T"] --> RET["在同任务子集内按 CLIP 特征余弦相似度检索最相似源物体"]
    DEMO --> RET
    RET --> STA["静态 affordance 迁移<br/>SD-DINOv2 逐像素特征匹配（argmax）→ 目标 2D 接触点 → 反投影为 3D 接触点 c3D"]
    STA --> DYN["动态 affordance 迁移<br/>平移 t = 源/目标 3D 接触点之差<br/>Point-SAM 以 c3D 为 prompt 分割可操作部件 → ICP 配准得旋转 R"]
    DYN --> TAU["目标轨迹 τT = R·τS + t"]
    subgraph POL["Affordance 引导的扩散策略（基于 DDIM）"]
        OT["场景点云 Ot：裁剪降采样 + MLP"] --> FT
        ST["本体感知 St：MLP"] --> FT
        STA --> FC["接触点 c：MLP"]
        TAU --> FTR["轨迹 τ：带 CLS token 的 Transformer 编码器"]
        FC --> FT["条件特征 ft = fO, fS, fΦ"]
        FTR --> FT
        FT --> EPS["噪声预测网络 εθ 迭代去噪"]
        EPS --> ACT["动作 a0"]
        GUIDE["推理时自适应 affordance 引导<br/>末端位置与 c3D 距离小于阈值 θ 时才施加梯度修正"] --> EPS
    end
```

图依据：论文 §3（Fig.3，affordance 记忆检索与静/动态迁移）、§4.1-4.3（条件扩散模型与 affordance 引导采样）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2025 |
| 方法名（清单标注） | AffordDP |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: explicit；Action Subcategory: Integrated policy · Explicit input |
| 论文 | <https://arxiv.org/abs/2412.03142> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（仅画论文方法部分明确写出的模块与数据流，不含推测），训练细节与实验请读 [原文](https://arxiv.org/abs/2412.03142)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2412.03142) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordDP」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_afforddp-generalizable-diffusion-policy-with-transferabl.md`](../../sources/papers/affordance_awesome_afforddp-generalizable-diffusion-policy-with-transferabl.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2412.03142>
- [arXiv PDF](https://arxiv.org/pdf/2412.03142)：已通读方法部分（实验与附录以略读为准）：§3.1-3.3（Affordance memory、静态/动态迁移）、§4.1-4.3（条件、训练、引导采样）；§5 实验与附录略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2412.03142)
