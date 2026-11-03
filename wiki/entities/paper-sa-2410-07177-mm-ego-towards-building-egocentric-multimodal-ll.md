---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2410.07177"
venue: "ICLR 2025"
summary: "Proposes MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2410_07177_mm-ego-towards-building-egocentric-multi.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# MM-Ego

**MM-Ego: Towards Building Egocentric Multimodal LLMs for Video QA** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 188/249** 篇，分组 **Advanced Reasoning & Cutting-edge Advances**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Proposes MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DE["数据引擎：narration 到 egocentric QA"]
        N["Ego4D 视频片段及其人工 narration，按时间顺序拼成 narration 段落"] --> GPT["GPT-4o（纯文本）生成 episodic memory 相关 QA，并标注所依据的 narration 句序号"]
        GPT --> MAP["句序号映射回视频时间帧，得到关键帧标签"]
    end
    V["视频均匀采样 N 帧"] --> SIG["SigLIP-so400m 视觉编码 + 2 层 MLP + 平均池化，得到高分辨率视觉嵌入"]
    SIG --> CMP["沿嵌入长度平均池化，得到压缩视觉嵌入"]
    Q["问题"] --> QE["Qwen2 tokenizer 与 embedding 层，得到问题嵌入"]
    CMP --> GG["Step 1 Global Glimpse：拼接 [压缩视觉嵌入, 问题嵌入, memory pointer 嵌入] 输入 LLM"]
    QE --> GG
    GG --> SC["末层输出做 softmax 点积，得到各帧相关性分数 s"]
    SC --> TK["叠加均匀采样扰动 u（探索与利用平衡），取 top-k 索引"]
    TK --> FB["Step 2 Fallback：取所选帧的高分辨率视觉嵌入 + 问题嵌入"]
    SIG --> FB
    QE --> FB
    FB --> ANS["LLM 生成回答"]
    MAP -. "训练：关键帧标签对 s 做二元交叉熵监督" .-> SC
```

图依据：论文 §2.1 数据引擎（Fig.2）、§2.2.1 嵌入、§2.2.2 Memory Pointer Prompting（Fig.4、式 (1)–(3)）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 188/249 |
| 分组 | Advanced Reasoning & Cutting-edge Advances |
| 出处 | ICLR 2025 |
| 论文 | <https://arxiv.org/abs/2410.07177> |
| 项目页 | <https://openreview.net/forum?id=7k9a8b7s9d> |

## 核心机制（归纳）

### 策展导读要点

Proposes MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.07177) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Advanced Reasoning & Cutting-edge Advances** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MM-Ego」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Advanced Reasoning & Cutting-edge Advances**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2410_07177_mm-ego-towards-building-egocentric-multi.md`](../../sources/papers/sun_awesome_ego_2410_07177_mm-ego-towards-building-egocentric-multi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2410.07177>
- [arXiv PDF](https://arxiv.org/pdf/2410.07177)：已通读方法部分（实验与附录以略读为准）：§2.1 数据引擎、§2.2.1 视觉与文本嵌入、§2.2.2 Memory Pointer Prompting

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2410.07177)
