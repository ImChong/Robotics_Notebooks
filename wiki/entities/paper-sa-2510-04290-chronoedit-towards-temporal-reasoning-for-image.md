---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2510.04290"
venue: "arXiv 2025"
code: https://github.com/nv-tlabs/ChronoEdit
summary: "Temporal reasoning for image editing and world simulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2510_04290_chronoedit-towards-temporal-reasoning-fo.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ChronoEdit

**ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 010/571** 篇，分组 **102 Post-Training & Test-Time Scaling**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Temporal reasoning for image editing and world simulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Temporal reasoning for image editing and world simulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TR["训练"]
        A1["公开图像编辑数据对 (c, p, y) 视为两帧视频"] --> A3
        A2["1.4M 合成视频及 VLM 生成的编辑指令：首帧为 c，末帧为 p，中间帧为 reasoning tokens"] --> A3
        A3["联合训练：以 Wan2.1-I2V-14B 为基础微调，rectified flow 速度预测"]
    end
    subgraph INF["推理"]
        IN["输入：参考图像 c 与编辑指令 y"] --> V["Wan2.1 VAE 编码 c 为首帧 latent；RoPE 把 c 锚定在时间步 0，输出 p 锚定在时间步 T"]
        V --> TRS["时间推理阶段：clean c 拼接带噪 reasoning tokens 与带噪目标帧，先去噪 Nr 步"]
        TRS --> DROP["丢弃 reasoning tokens，仅保留部分去噪的目标帧 latent"]
        DROP --> GEN["编辑帧生成阶段：clean c 拼接目标 latent（重复 4 帧），完成剩余 N-Nr 步去噪"]
        GEN --> DEC["VAE 解码，取最后一帧作为编辑结果"]
    end
    A3 -.->|"训练后的 ChronoEdit"| V
    DMD["可选：DMD 蒸馏得到 8 步学生模型，加速推理"] -.-> GEN
```

图依据：论文 §3.2 / §3.3 / §3.4、Fig.3 与 Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 010/571 |
| 分组 | 102 Post-Training & Test-Time Scaling |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.04290> |
| 代码/项目 | <https://github.com/nv-tlabs/ChronoEdit> |

## 核心机制（归纳）

### 策展导读要点

Temporal reasoning for image editing and world simulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.04290) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **102 Post-Training & Test-Time Scaling** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ChronoEdit」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **102 Post-Training & Test-Time Scaling**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2510_04290_chronoedit-towards-temporal-reasoning-fo.md`](../../sources/papers/sun_awesome_wm_2510_04290_chronoedit-towards-temporal-reasoning-fo.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2510.04290>
- [arXiv PDF](https://arxiv.org/pdf/2510.04290)：已通读方法部分（实验与附录以略读为准）：§3.1–§3.4（含 Fig.3、Algorithm 1）；§4 实验仅略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2510.04290)
