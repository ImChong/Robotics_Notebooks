---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.06216"
venue: "arXiv 2026"
summary: "First NPU-based real-time interactive world model achieving 50 FPS with 30%-50% inference cost."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_06216_moworld-a-flash-world-model.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# MoWorld

**MoWorld: A Flash World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 163/571** 篇，分组 **51 General Interactive Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

First NPU-based real-time interactive world model achieving 50 FPS with 30%-50% inference cost.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- First NPU-based real-time interactive world model achieving 50 FPS with 30%-50% inference cost.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["数据引擎"]
        D1["数据收集<br/>开放域视频 + 游戏场景数据"] --> D2["几何补全与质量控制<br/>恢复缺失相机条件，剔除几何/视觉不可靠样本"]
        D2 --> D3["VLM 标注<br/>场景级描述，文本不含相机运动"]
        D3 --> D4["离线预计算缓存<br/>视频潜变量、文本嵌入、相机条件"]
    end
    D4 --> PRE
    subgraph PRE["预训练（NPU 集群）"]
        P1["条件：首帧 x0 + 文本 + 相机轨迹<br/>相机内外参 → Plücker 射线 → Camera Adapter → 加到 patch token"] --> P2["高噪/低噪 MoE DiT + camera adapter<br/>课程式训练：短 clip 125/250F → 中 500/1000F → 长 2000F"]
    end
    P2 -- "预训练模型作 teacher" --> AR1
    subgraph DIST["自回归蒸馏"]
        AR1["AR Flow Matching 预训练<br/>真值前缀 History + 锚点帧，仅监督当前 chunk"] --> AR2["Self-Forcing / DMD 式蒸馏<br/>学生真实 AR rollout；Fake Phase 与 Student Phase 交替<br/>冻结 teacher 与 Fake Model 给出分布匹配梯度"]
        AR2 --> AR3["少步因果学生（采样步数 50 → 4）"]
    end
    AR3 --> INF
    subgraph INF["实时推理"]
        I1["首帧 + 文本 + 用户相机控制 W/A/S/D<br/>编码为初始条件，随后释放编码器权重"] --> I2["History Bank 选帧：最近帧 + 初始帧 + 相机相关历史帧<br/>与当前 chunk 拼接送入 DiT"]
        I2 --> I3["高噪 → 低噪专家依次去噪，预测未来帧潜变量"]
        I3 --> I4["VAE 解码并流式显示"]
        I3 -- "写回 History Bank" --> I2
    end
```

图依据：论文 §2（Fig.2 数据引擎）、§3.1–3.3（预训练）、§4.2–4.4（Fig.4 蒸馏）、§5.2（Fig.6 推理流程）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 163/571 |
| 分组 | 51 General Interactive Frameworks |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.06216> |

## 核心机制（归纳）

### 策展导读要点

First NPU-based real-time interactive world model achieving 50 FPS with 30%-50% inference cost.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.06216) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **51 General Interactive Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MoWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **51 General Interactive Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2607_06216_moworld-a-flash-world-model.md`](../../sources/papers/sun_awesome_wm_2607_06216_moworld-a-flash-world-model.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.06216>
- [arXiv PDF](https://arxiv.org/pdf/2607.06216)：已通读方法部分（实验与附录以略读为准）：摘要、§2 数据引擎、§3 模型预训练、§4 自回归蒸馏、§5 实时推理（5.1–5.4）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.06216)
