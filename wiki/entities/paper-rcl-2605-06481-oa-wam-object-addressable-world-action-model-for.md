---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2605.06481"
summary: "We propose OA-WAM, an Object-Addressable World Action Model for robust robot manipulation. OA-WAM matches strong VLA and WAM baselines on LIBERO (97.8%) and SimplerEnv (79.3%), reaches state-of-the-art performance on the"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2605_06481_oa-wam-object-addressable-world-action-m.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# OA-WAM

**OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 442/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We propose OA-WAM, an Object-Addressable World Action Model for robust robot manipulation. OA-WAM matches strong VLA and WAM baselines on LIBERO (97.8%) and SimplerEnv (79.3%), reaches state-of-the-art performance on the most relevant LIBERO-Plus geometric axes, and remains competitive on the seven-axis aggregate.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We propose OA-WAM, an Object-Addressable World Action Model for robust robot manipulation. OA-WAM matches strong VLA and WAM baselines on LIBERO (97.8%) and SimplerEnv (79.3%), reaches state-of-the-art performance on the most relevant LIBERO-Plus geometric axes, and remains competitive on the seven-axis aggregate.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：第三人称 + 腕部 RGB It、本体 qt、语言指令 ℓ、过去动作 a<t"] --> P["冻结的基础感知产生六路 token 流<br/>BPE 文本 / Qwen3-VL 名词短语（仅作 SAM 3 提示）/ Chameleon VQ-GAN 图像码<br/>SAM 3 + DINOv3 + 位姿对象 slot / 本体离散化 / 过去动作离散化"]
    P --> SL["每帧 N+1 个 slot（1 个机器人 + N 个物体）<br/>slot = addr(32 维，t=0 算出后固定) ‖ content(256 维，逐帧更新) ‖ 帧序号嵌入 ‖ 角色标签"]
    SL --> AD["slot adapter 投影到 4096 维<br/>替换序列中的 slot 占位位置"]
    AD --> SEQ["block-causal 序列：文本 / 状态 / 各帧 VQ-Slot-动作 / 末尾 [ACT_Q]"]
    SEQ --> TR["Chameleon-7B trunk（32 层）<br/>slot 位置的 key 只取 addr 子向量，每层后重置 addr 切片"]
    TR --> WH["World head：预测下一帧各物体 slot 的 content 与位姿<br/>MSE，机器人 slot 不计入"]
    TR --> AH["Action head：读取 [ACT_Q] 隐状态<br/>flow matching，4 步 Euler 一次前向解出 16 步动作 chunk"]
    TR --> VQ["辅助 VQ head：预测下一帧 VQ token"]
```

图依据：论文 §3.1–3.5（问题设定、slot 化与统一序列、Object-addressable attention、预测头与训练目标）/ Fig. 3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 442/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2605.06481> |
| 子类 / 象限 | 潜空间预测与JEPA · 泛化与动作对齐 · Q1 · One Model × 联合预测 |

## 核心机制（归纳）

### 策展导读要点

We propose OA-WAM, an Object-Addressable World Action Model for robust robot manipulation. OA-WAM matches strong VLA and WAM baselines on LIBERO (97.8%) and SimplerEnv (79.3%), reaches state-of-the-art performance on the most relevant LIBERO-Plus geometric axes, and remains competitive on the seven-axis aggregate.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.06481) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「OA-WAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2605_06481_oa-wam-object-addressable-world-action-m.md`](../../sources/papers/rcl_awesome_wam_2605_06481_oa-wam-object-addressable-world-action-m.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2605.06481>
- [arXiv PDF](https://arxiv.org/pdf/2605.06481)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 相关工作、§3 方法（3.1–3.5）、§4.1 实验设置

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2605.06481)
