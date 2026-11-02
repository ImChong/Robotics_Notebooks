---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.02577"
venue: "arXiv 2026"
summary: "Generalizable embodiment-centric world model for scalable data generation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_02577_robodream-compositional-world-models-for.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# RoboDream

**RoboDream: Compositional World Models for Scalable Robot Data Synthesis** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 226/571** 篇，分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generalizable embodiment-centric world model for scalable data generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Generalizable embodiment-centric world model for scalable data generation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph PRIOR["训练数据：自动提取先验（论文 §III-C）"]
        F1["机器人数据集首帧 o1 + 任务指令 ℓ"] --> VLM["GPT-5-nano 识别任务相关物体"]
        VLM --> GS["Grounded-SAM 分割物体"]
        GS --> OP["物体先验 Io：裁剪、随机旋转缩放<br/>放到空白画布"]
        GS --> OMNI["OmniPaint 修复去除物体后的空洞"]
        OMNI --> SP["场景先验 Is：干净背景图"]
    end
    subgraph MODE["运动来源：两种部署模式"]
        RR["Retrieval and Rebirth<br/>T5 嵌入指令，余弦相似度检索已有轨迹<br/>Isaac Lab 重放并可换新视角"]
        PF["Prop-Free 遥操作<br/>无真实物体的 pantomime 动作<br/>现实空场景或仿真中采集"]
        RR --> VR["渲染的纯机器人运动视频 vrob"]
        PF --> VR
    end
    subgraph DIT["多模态视频扩散 Transformer"]
        CH["通道拼接：带噪视频 latent zt<br/>+ VAE(vrob) + VAE(Is 沿时间扩展)"]
        OS["物体先验 token 经扩展 self-attention 注入"]
        CA["任务指令（T5）与全局轨迹 τ（MLP）<br/>经 cross-attention 注入"]
        CH --> TF["Transformer 去噪"]
        OS --> TF
        CA --> TF
    end
    VR --> CH
    SP --> CH
    OP --> OS
    TF --> OUT["生成机器人演示视频 o1:T<br/>（多视角：各视角独立 tokenize 后堆叠 token）"]
    OUT --> POL["训练下游策略<br/>实验中使用 Diffusion Policy"]
```

图依据：论文 §III-A/B/C/D、§IV-A / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 226/571 |
| 分组 | 63 World Models for VLA Training & Evaluation Combining Paradigms |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.02577> |

## 核心机制（归纳）

### 策展导读要点

Generalizable embodiment-centric world model for scalable data generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.02577) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **63 World Models for VLA Training & Evaluation Combining Paradigms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RoboDream」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_02577_robodream-compositional-world-models-for.md`](../../sources/papers/sun_awesome_wm_2606_02577_robodream-compositional-world-models-for.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.02577>
- [arXiv PDF](https://arxiv.org/pdf/2606.02577)：已通读方法部分（实验与附录以略读为准）：§I–II 引言与相关工作（略读）、§III 方法（A–D）、§IV-A/B 实验设定；其余略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.02577)
