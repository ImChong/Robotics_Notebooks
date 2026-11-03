---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2506.14198"
venue: "2025"
summary: "AMPLIFY learns a compact vocabulary of visual motion from point tracks, predicts that motion from an image and task instruction, and translates it into robot actions through a separate inverse model. Its strongest eviden"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2506_14198_amplify-actionless-motion-priors-for-rob.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# AMPLIFY

**AMPLIFY: Actionless Motion Priors for Robot Learning from Videos** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 278/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AMPLIFY learns a compact vocabulary of visual motion from point tracks, predicts that motion from an image and task instruction, and translates it into robot actions through a separate inverse model. Its strongest evidence concerns scarce target-task action labels, including transfer where target-task videos remain...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- AMPLIFY learns a compact vocabulary of visual motion from point tracks, predicts that motion from an image and task instruction, and translates it into robot actions through a separate inverse model. Its strongest evidence concerns scarce target-task action labels, including transfer where target-task videos remain...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["视频数据 V（含无动作的人类视频）"] --> TR["现成点跟踪模型：20×20 网格 N=400 个点，跟踪 T=16 帧"]
    TR --> VEL["单步速度 u_t"]
    VEL --> TOK["阶段 1：运动 Tokenizer<br/>keypoint 编码器 → FSQ 离散量化 → keypoint 解码器<br/>（局部 W×W 窗口分类，交叉熵重建）"]
    TOK -- "冻结，提供 z_t = E(u_t)" --> FWD
    V --> FWD["阶段 2：前向动力学 f（自回归 Transformer）<br/>输入：ResNet-18 图像 token + T5 语言 token + SOS<br/>输出：潜在运动 token ẑ_t（交叉熵）"]
    R["机器人交互数据 R：图像 o、本体状态 q、动作 a"] --> INV
    TOK -- "冻结" --> INV
    FWD -- "冻结，用其预测 ẑ_t 微调" --> INV["阶段 3：逆动力学 f_inv<br/>可学习 token 交叉注意力到图像/本体/运动码<br/>→ 动作头（高斯，折扣 NLL）→ 动作块 a_t"]
    subgraph INF["推理"]
        O["当前观测 o_t + 任务描述 g"] --> F2["f 预测潜在运动 ẑ_t"]
        F2 --> I2["f_inv 结合图像与本体状态解码动作块"]
        I2 --> TE["时间集成（temporal ensembling）→ 执行"]
    end
```

图依据：论文 §2 方法（§2.1 关键点轨迹预处理、§2.2 运动 Tokenization / Fig.2a、§2.3 前向动力学 / Fig.2b、§2.4 逆动力学 / Fig.2c、§2.5 推理）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 278/564 |
| 分组 | WAMs |
| 出处 | 2025 |
| 论文 | <https://arxiv.org/abs/2506.14198> |
| 子类 / 象限 | 视觉规划与IDM · Q4 · Dual-system × IDM |

## 核心机制（归纳）

### 策展导读要点

AMPLIFY learns a compact vocabulary of visual motion from point tracks, predicts that motion from an image and task instruction, and translates it into robot actions through a separate inverse model. Its strongest evidence concerns scarce target-task action labels, including transfer where target-task videos remain...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.14198) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AMPLIFY」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2506_14198_amplify-actionless-motion-priors-for-rob.md`](../../sources/papers/rcl_awesome_wam_2506_14198_amplify-actionless-motion-priors-for-rob.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2506.14198>
- [arXiv PDF](https://arxiv.org/pdf/2506.14198)：已通读方法部分（实验与附录以略读为准）：§2.1–2.5 方法；§3 实验仅略读。

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2506.14198)
