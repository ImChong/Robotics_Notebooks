---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2406.09394"
venue: "arXiv 2024"
summary: "Interactive 3D scene generation from a single image."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2406_09394_wonderworld-interactive-3d-scene-generat.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WonderWorld

**WonderWorld: Interactive 3D Scene Generation from a Single Image** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 009/571** 篇，分组 **101 Memory Mechanisms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Interactive 3D scene generation from a single image.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Interactive 3D scene generation from a single image.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["初始图像 I0"] --> B["VLM 描述图像<br/>生成首个场景 FLAGS"]
    B --> C
    subgraph LOOP["交互生成循环 Alg.1 / Fig.2"]
        C["用户移动相机 Cgen<br/>并输入文本提示 U"] --> D["LLM 生成结构化描述<br/>前景 F 背景 B 风格 S"]
        D --> E["渲染已有几何 并做文本引导外推<br/>得到新场景图像"]
        E --> F["引导深度扩散 §3.2<br/>以已渲染深度为引导估计新深度"]
        F --> G["单视角分层生成 §3.1<br/>前景 背景 天空 三层图像与掩码<br/>被遮挡层用扩散 inpaint 补全"]
        G --> H["几何初始化<br/>像素对齐位置 法线 尺度"]
        H --> I["FLAGS 优化<br/>由后向前 天空 背景 前景 逐层 100 步 Adam"]
        I --> J["并入已有场景<br/>实时渲染 用户继续移动相机"]
        J --> C
    end
```

图依据：论文 §3 / §3.1–§3.2 / Fig.2、Fig.3、Fig.4；外层循环据附录 Alg.1 与 Fig.2 图注（Alg.1 仅略读）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 009/571 |
| 分组 | 101 Memory Mechanisms |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2406.09394> |
| 项目页 | <https://kovenyu.com/wonderworld/> |

## 核心机制（归纳）

### 策展导读要点

Interactive 3D scene generation from a single image.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2406.09394) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **101 Memory Mechanisms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WonderWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **101 Memory Mechanisms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2406_09394_wonderworld-interactive-3d-scene-generat.md`](../../sources/papers/sun_awesome_wm_2406_09394_wonderworld-interactive-3d-scene-generat.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2406.09394>
- [arXiv PDF](https://arxiv.org/pdf/2406.09394)：已通读方法部分（实验与附录以略读为准）：§3 方法概述、§3.1 FLAGS（定义、单视角分层生成、几何初始化、优化）、§3.2 引导深度扩散；附录 Alg.1 略读，§4 实验与其余附录以略读/未读为准


## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2406.09394)
