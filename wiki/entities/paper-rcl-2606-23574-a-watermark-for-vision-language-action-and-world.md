---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2606.23574"
code: https://github.com/Y-L-LIU/keyed-latent-watermarking-vla
summary: "To address this, we propose the \\emph{keyed latent-provenance verification} method, which fingerprints the policy through the seed of the Gaussian noise vector that the models draw before generation."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2606_23574_a-watermark-for-vision-language-action-a.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# A Watermark for Vision-Language-Action and World Action Models

**A Watermark for Vision-Language-Action and World Action Models** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 193/564** 篇，分组 **Related resources**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

To address this, we propose the \emph{keyed latent-provenance verification} method, which fingerprints the policy through the seed of the Gaussian noise vector that the models draw before generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- To address this, we propose the \emph{keyed latent-provenance verification} method, which fingerprints the policy through the seed of the Gaussian noise vector that the models draw before generation.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph INJ["注入 (owner 侧)"]
    K["密钥 k* + episode nonce ν_e"] --> SEL["密钥选择器 S(k*, ν_e, c)<br/>每 episode 至多标记 m 个 chunk"]
    SEL -->|"选中 chunk"| PERT["密钥潜变量扰动<br/>z_fp = sqrt(1-β²)·z_c + β·r_c<br/>保持 N(0, I) 噪声分布"]
    SEL -->|"未选中"| BASE["使用原始种子 z_c"]
    PERT --> GEN["生成器 F_θ (flow matching / diffusion)<br/>VLA 或 world-action model"]
    BASE --> GEN
    GEN --> ACT["动作 chunk"]
  end
  ACT --> POST["部署侧后处理 (被审计方)<br/>clip / EMA / jitter / delay<br/>且仅可见 D_env 个执行通道"]
  subgraph VER["验证 (owner 审计)"]
    POST --> MAP["MAP 潜变量恢复<br/>由执行动作估计种子 ẑ_c"]
    MAP --> SC["对候选密钥匹配滤波打分 s_e(k)<br/>(含同步滞后搜索)"]
    SC --> Z["诱饵密钥 (decoy) 归一化<br/>得到 Z-score Z_e(k)"]
    Z --> AGG["多 rollout 累加 T_G(k)"]
    AGG --> DEC{"T_G(k*) 超过诱饵分布<br/>(1-α_FPR) 分位阈值?"}
  end
```

图依据：论文 §3 威胁模型、§4.1 指纹注入、§4.2 指纹验证

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 193/564 |
| 分组 | Related resources |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2606.23574> |
| 代码/项目 | <https://github.com/Y-L-LIU/keyed-latent-watermarking-vla> |
| 子类 / 象限 | 评测协议与诊断 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

To address this, we propose the \emph{keyed latent-provenance verification} method, which fingerprints the policy through the seed of the Gaussian noise vector that the models draw before generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.23574) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Related resources** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「A Watermark for Vision-Language-Action and World Action Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Related resources**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2606_23574_a-watermark-for-vision-language-action-a.md`](../../sources/papers/rcl_awesome_wam_2606_23574_a-watermark-for-vision-language-action-a.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2606.23574>
- [arXiv PDF](https://arxiv.org/pdf/2606.23574)：已通读方法部分（实验与附录以略读为准）：§3 威胁模型与问题设定、§4 方法（4.1–4.2）；§5 以后（评估、攻击实验）略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2606.23574)
