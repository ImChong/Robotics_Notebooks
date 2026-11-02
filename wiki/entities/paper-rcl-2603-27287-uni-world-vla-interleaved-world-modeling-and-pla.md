---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2603.27287"
venue: "2026"
summary: "RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2603_27287_uni-world-vla-interleaved-world-modeling.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Uni-World VLA

**Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 503/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    H["历史第一人称帧 I(t-M..t-1)"] --> VQ["MagVIT-v2 编码为离散 token<br/>上下文 token c（高分辨率）+ 动态 token d（10 Hz、低分辨率）"]
    H --> DA["Depth Anything 3 估计深度图 D<br/>缩放为两种分辨率"]
    DA --> DE["深度编码器 CDE / DDE"]
    VQ --> CA["交叉注意力融合<br/>token 嵌入为 query，深度特征为 key/value"]
    DE --> CA
    EGO["自车状态：速度、加速度、高层驾驶指令"] --> IN
    PR["系统提示 + 用户提示<br/>要求规划前先想象近未来"] --> IN
    CA --> IN["拼接输入序列<br/>[系统提示 | 动态与上下文 token | 用户提示 | Ego token]"]
    IN --> LLM["Show-o（Phi-1.5 多模态 LLM）自回归生成"]
    LLM --> FT["生成第 t+k 帧视觉 token"]
    FT -->|"追加进上下文，输入 action query"| AQ["生成第 t+k 步动作 token"]
    AQ -->|"k+1，重复至 N=8 步（4 秒）"| FT
    FT --> DEC["MagVIT-v2 解码器 + 上下文 token<br/>得到未来 RGB 帧"]
    AQ --> MLP["MLP 头回归自车位置<br/>组成规划轨迹"]
```

图依据：论文 §3（Methods）/ Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 503/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2603.27287> |
| 子类 / 象限 | 视觉规划与IDM · 自动驾驶 · Q2 · One Model × IDM |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.27287) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Uni-World VLA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2603_27287_uni-world-vla-interleaved-world-modeling.md`](../../sources/papers/rcl_awesome_wam_2603_27287_uni-world-vla-interleaved-world-modeling.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2603.27287>
- [arXiv PDF](https://arxiv.org/pdf/2603.27287)：已通读方法部分（实验与附录以略读为准）：§1–3（Inputs and tokenization、Interleaved frame-action generation、Decoding、训练目标与注意力掩码、推理、Depth integration）；§4 实验与附录仅略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2603.27287)
