---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2510.08713"
venue: "arXiv 2025"
code: https://github.com/F1y1113/UniWM
summary: "Memory-augmented planning and foresight for visual navigation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2510_08713_unified-world-models-memory-augmented-pl.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Unified World Models

**Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 356/571** 篇，分组 **813 Navigation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Memory-augmented planning and foresight for visual navigation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Memory-augmented planning and foresight for visual navigation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TR["训练"]
        T1["轨迹滑窗采样：规划样本与世界模型样本在同一 batch 内交错"] --> T2["三类分词：VQ 图像 token、BPE 文本与位姿 token、bin 动作 token"]
        T2 --> T3["统一自回归 MLLM Fθ（基于 Chameleon / Anole），分词器冻结，仅更新 Transformer"]
        T3 --> T4["损失：规划用离散 bin token 损失 L_plan，世界模型用重建损失 L_world"]
    end
    subgraph INF["推理（每一步 t）"]
        I1["输入：起点观测 o_s、目标观测 o_g、初始位姿 p0、当前观测"] --> I2["重置 intra-step 记忆，从当前观测 token 段的选定解码层抽取 K、V"]
        I2 --> I3["时空融合：与 cross-step 记忆库做 top-k 余弦相似度门控，加指数时间衰减，得到融合记忆"]
        I3 --> I4["子步 A 规划：Fθ 预测下一动作"]
        I4 -->|"输出 Stop"| I7["结束导航"]
        I4 --> I5["子步 B 世界模型：Fθ 预测执行动作后的下一观测"]
        I5 --> I6["当前 intra-step 记忆追加进 cross-step 记忆库"]
        I6 --> I2
    end
```

图依据：论文 §3.1–§3.3 / Fig.2 / Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 356/571 |
| 分组 | 813 Navigation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.08713> |
| 代码/项目 | <https://github.com/F1y1113/UniWM> |

## 核心机制（归纳）

### 策展导读要点

Memory-augmented planning and foresight for visual navigation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.08713) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **813 Navigation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Unified World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **813 Navigation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2510_08713_unified-world-models-memory-augmented-pl.md`](../../sources/papers/sun_awesome_wm_2510_08713_unified-world-models-memory-augmented-pl.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2510.08713>
- [arXiv PDF](https://arxiv.org/pdf/2510.08713)：已通读方法部分（实验与附录以略读为准）：§3.1 预备与统一表述、§3.2 统一训练方案、§3.3 记忆库推理（含 Fig.2、Fig.3）；§4 实验未读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2510.08713)
