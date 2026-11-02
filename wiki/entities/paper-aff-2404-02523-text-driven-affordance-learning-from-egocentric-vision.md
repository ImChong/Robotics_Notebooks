---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-text-driven-affordance-learning-from-egocentric-vision
arxiv: "2404.02523"
venue: "arXiv 2024"
summary: "arXiv 2024 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Interaction-driven）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_text-driven-affordance-learning-from-egocentric-vision.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Text-driven Affordance Learning from Egocentric Vision

**Text-driven Affordance Learning from Egocentric Vision**（arXiv 2024）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Text-driven Affordance Learning from Egocentric Vision——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Interaction-driven）**（方法流程见下方「流程总览」）。

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
    V["第一人称视频（Ego4D / Epic-Kitchens）<br/>动作描述 + 起止时间戳 → Fobs / Finter"] --> IC["交互分类：LLaMA2-70B 判断 hand-object 还是 tool-object<br/>（拼接前两条描述，system prompt + few-shot），并取工具名"]
    IC --> SEG["开放词表分割：在 Finter 上用 hand 或工具名得分割图<br/>与手-物检测器的物体框求交，得接触区域"]
    V --> H["手-物检测器给出框作掩码；SURF 对应点 + RANSAC 估计相邻帧 homography"]
    SEG --> PC["接触区域经 homography 投影到 Fobs<br/>拟合 GMM → 采样点 + Gaussian blur 得接触点热图"]
    H --> PC
    V --> TR["CoTracker 稠密点追踪 (tinter 到 tinter+1.0 秒)<br/>homography 投影到 Fobs 得轨迹 P"]
    H --> TR
    TR --> FIT["曲线回归：拟合旋转+平移的参数化轨迹 f(t)"]
    PC --> DS["TextAFF80K：图像 x / 动作描述 d / 接触点 p / 轨迹 t"]
    FIT --> DS
    DS --> NET["训练扩展后的 MDETR / CLIPSeg<br/>输入 x + d，输出接触点热图 + 轨迹函数参数"]
```

图依据：论文 §III / Fig.2（伪标签生成）、§III-B（网络结构）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2024 |
| 方法名（清单标注） | - |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Interaction-driven |
| 论文 | <https://arxiv.org/abs/2404.02523> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上方「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2404.02523)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2404.02523) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Text-driven Affordance Learning from Egocentric Vision」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_text-driven-affordance-learning-from-egocentric-vision.md`](../../sources/papers/affordance_awesome_text-driven-affordance-learning-from-egocentric-vision.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2404.02523>
- [arXiv PDF](https://arxiv.org/pdf/2404.02523)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 文本驱动 affordance 学习（伪标签生成与网络结构）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2404.02523)
