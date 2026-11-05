---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2308.12966"
venue: "2023"
code: https://github.com/QwenLM/Qwen-VL
summary: "RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2308_12966_qwen-vl-a-versatile-vision-language-mode.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Qwen-VL

**Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 085/564** 篇，分组 **Components of WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
    IMG["输入图像<br/>先缩放到固定分辨率"] --> VIT["视觉编码器 ViT<br/>以步长 14 切 patch<br/>得到图像特征序列"]
    VIT --> ADP["位置感知视觉语言适配器<br/>单层 cross-attention<br/>可训练 query 向量 + 2D 绝对位置编码"]
    ADP --> FEAT["压缩为定长 256 的图像特征<br/>首尾加 img 与 /img 标记"]
    TXT["文本 / 指令<br/>区域用 box 与 ref 标记"] --> LLM
    FEAT --> LLM["大语言模型 Qwen-7B 初始化"]
    LLM --> OUT["文本输出<br/>含归一化到 0-1000 的框坐标串"]
    subgraph TRAIN["三阶段训练"]
        direction LR
        S1["阶段 1 预训练<br/>224 分辨率 图文对<br/>冻结 LLM"] --> S2["阶段 2 多任务预训练<br/>448 分辨率 7 类任务<br/>全模型解冻"] --> S3["阶段 3 监督微调<br/>对话数据 约 35 万条<br/>冻结视觉编码器"]
    end
```

图依据：论文 §2.1 Model Architecture、§2.2 Inputs and Outputs、§3 Training / Fig.3（三阶段训练流程）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 085/564 |
| 分组 | Components of WAMs |
| 出处 | 2023 |
| 论文 | <https://arxiv.org/abs/2308.12966> |
| 代码/项目 | <https://github.com/QwenLM/Qwen-VL> |
| 子类 / 象限 | Language & vision-language backbones · 语言与VLM Backbone · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2308.12966) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Components of WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Qwen-VL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Components of WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2308_12966_qwen-vl-a-versatile-vision-language-mode.md`](../../sources/papers/rcl_awesome_wam_2308_12966_qwen-vl-a-versatile-vision-language-mode.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2308.12966>
- [arXiv PDF](https://arxiv.org/pdf/2308.12966)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 Methodology（模型结构、输入输出）、§3 Training（三阶段）；§4 评测与附录仅略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2308.12966)
