---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2411.18179"
venue: "NeurIPS 2024"
summary: "RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2411_18179_prediction-with-action-visual-policy-lea.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Prediction with Action

**Prediction with Action: Visual Policy Learning via Joint Denoising Process** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 453/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

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
    OBS["当前观测：RGB 图像 c_I、机器人位姿 c_A<br/>(真机另有深度 c_E)"] --> ENC["各模态编码<br/>RGB：冻结 VAE；位姿：MLP；深度：下采样"]
    ENC --> CAT["条件潜变量与 k 步未来白噪声<br/>按通道拼接"]
    CAT --> TOK["tokenize 为 t_I, t_A, t_E<br/>拼接成统一 token 序列"]
    LANG["语言指令 l → 冻结 CLIP 文本嵌入"] --> DIT
    TOK --> DIT["DiT 块 ×N 联合去噪<br/>masked attention 屏蔽缺失模态"]
    DIT --> OUT["同时输出未来 k 步图像 x_I、动作 x_A (及深度 x_E)"]
    OUT --> EXE["只执行首个动作 x_A^1<br/>再触发新一轮预测 (闭环)"]
    EXE -.-> OBS
    PRE["ImageNet 预训练 DiT 初始化<br/>新增层零初始化"] --> DIT
    LOSS["训练：三模态 DDPM 损失加权 L = λI·L_I + λA·L_A + λE·L_E"] --> DIT
```

图依据：论文 §3.1 Overview / §3.2 Model Architectures 与 Fig.3 / §3.3 Training Process

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 453/564 |
| 分组 | WAMs |
| 出处 | NeurIPS 2024 |
| 论文 | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbe25fa0e7c7084049276888a09acc8d-Abstract-Conference.html> |
| 子类 / 象限 | 联合视频动作建模 · Q1 · One Model × 联合预测 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbe25fa0e7c7084049276888a09acc8d-Abstract-Conference.html) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Prediction with Action」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2411_18179_prediction-with-action-visual-policy-lea.md`](../../sources/papers/rcl_awesome_wam_2411_18179_prediction-with-action-visual-policy-lea.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbe25fa0e7c7084049276888a09acc8d-Abstract-Conference.html>
- [arXiv PDF](https://arxiv.org/pdf/2411.18179)：已通读方法部分（实验与附录以略读为准）：§3 PAD（3.1 至 3.3）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbe25fa0e7c7084049276888a09acc8d-Abstract-Conference.html)
