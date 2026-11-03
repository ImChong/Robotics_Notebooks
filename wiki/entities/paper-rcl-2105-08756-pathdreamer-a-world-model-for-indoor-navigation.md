---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2105.08756"
venue: "ICCV 2021"
code: https://github.com/google-research/pathdreamer
summary: "RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2105_08756_pathdreamer-a-world-model-for-indoor-nav.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Pathdreamer

**Pathdreamer: A World Model for Indoor Navigation** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 448/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

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
  HIST["历史观测 1..t-1<br/>RGB / 深度 / 语义（360° 全景）"] --> PC["累积 3D 点云<br/>由深度 + 相机位姿反投影"]
  POSE["未来视点位姿 T_t"] --> PROJ
  PC --> PROJ["按 T_t 重投影<br/>稀疏语义 / 深度 / RGB 引导图"]
  subgraph S1["阶段 1：Structure Generator（随机编码器-解码器）"]
    ENC["编码器（RedNet / ResNet-50）"]
    NOISE["噪声张量 z_t<br/>推理时采样先验 CNN_prior，训练时用后验 CNN_post"]
    DEC["解码器<br/>输出预测语义 ŝ_t + 深度 d̂_t"]
    ENC --> NOISE --> DEC
  end
  PROJ -->|稀疏语义 / 深度引导图| ENC
  DEC -->|"反投影并加入点云"| PC
  subgraph S2["阶段 2：Image Generator（Multi-SPADE GAN）"]
    GEN["1 个卷积块 + 7 个 Multi-SPADE 残差块<br/>分别以 [ŝ_t, d̂_t] 与 RGB 引导图为条件"]
  end
  DEC --> GEN
  PROJ -->|稀疏 RGB 引导图| GEN
  GEN --> OUT["预测 RGB 图像 Î_t"]
  OUT -.->|作为下一步历史观测| HIST
```

图依据：论文 §3 Pathdreamer、Fig. 2（两阶段架构）、§3.1 Structure Generator、§3.2 Image Generator、§3.3 训练与推理；两阶段独立训练，图仅画推理时的数据流。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 448/564 |
| 分组 | WAMs |
| 出处 | ICCV 2021 |
| 论文 | <https://openaccess.thecvf.com/content/ICCV2021/html/Koh_Pathdreamer_A_World_Model_for_Indoor_Navigation_ICCV_2021_paper.html> |
| 代码/项目 | <https://github.com/google-research/pathdreamer> |
| 子类 / 象限 | 导航 · 记忆与长时序 · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://openaccess.thecvf.com/content/ICCV2021/html/Koh_Pathdreamer_A_World_Model_for_Indoor_Navigation_ICCV_2021_paper.html) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Pathdreamer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2105_08756_pathdreamer-a-world-model-for-indoor-nav.md`](../../sources/papers/rcl_awesome_wam_2105_08756_pathdreamer-a-world-model-for-indoor-nav.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://openaccess.thecvf.com/content/ICCV2021/html/Koh_Pathdreamer_A_World_Model_for_Indoor_Navigation_ICCV_2021_paper.html>
- [arXiv PDF](https://arxiv.org/pdf/2105.08756)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Pathdreamer（含 3.1–3.3）、§4 实验开头；附录未读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://openaccess.thecvf.com/content/ICCV2021/html/Koh_Pathdreamer_A_World_Model_for_Indoor_Navigation_ICCV_2021_paper.html)
