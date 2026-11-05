---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2506.13762"
venue: "arXiv 2025"
summary: "VisuoTactile Local policy learning; removing tactile input reduces success rates by average 40%."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2506_13762_touch-begins-where-vision-ends-generaliz.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Touch begins where vision ends

**Touch begins where vision ends: Generalizable policies for contact-rich manipulation (ViTaL)** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 028/065** 篇，分组 **51 Imitation Learning with Tactile Feedback**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

VisuoTactile Local policy learning; removing tactile input reduces success rates by average 40%.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- VisuoTactile Local policy learning; removing tactile input reduces success rates by average 40%.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph P1["阶段 1：视觉-触觉行为克隆"]
        D["VR 遥操作示教（绿幕背景，目标物位置固定，32 条/任务）<br/>观测：腕部相机 + 触觉；动作：末端坐标系相对动作"]
        ANN["人工在参考帧标关键点 → DIFT 对应匹配 → SAM2 分割 → XMem 跟踪"]
        AUG["RoboEngine 程序化生成背景，只替换任务无关区域"]
        BC["ResNet-18（图像）+ MLP（AnySkin 触觉）→ 视觉触觉 Transformer π_b<br/>动作分块，MSE 损失"]
        D --> ANN --> AUG --> BC
    end
    subgraph P2["阶段 2：残差强化学习"]
        FZ["冻结编码器与 π_b"]
        RES["残差策略 π_r：输入 (z_i, z_t, a_b) → 残差 a_r<br/>n-step DDPG；奖励 = 人工二值成功 + 目标 L1 距离；回放同样做背景增广"]
        FZ --> RES
    end
    BC --> FZ
    subgraph P3["推理：VLM 引导到达"]
        EXT["外部 RGB-D 观测 + 语言指定目标"] --> MOL["Molmo 预测目标 2D 点"]
        MOL --> PRJ["结合深度与标定投影到 3D，在目标附近采样初始末端位姿"]
        PRJ --> LOC["局部视觉触觉策略执行：a = a_b + a_r"]
    end
    RES --> LOC
    BC --> LOC
```

图依据：论文 §3 / Fig.2（ViTaL 三阶段总览）、§3.1（语义增广行为克隆）、§3.2（残差 RL）、§3.3（推理）。注：论文中该方法名为 ViTaL。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 028/065 |
| 分组 | 51 Imitation Learning with Tactile Feedback |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.13762> |

## 核心机制（归纳）

### 策展导读要点

VisuoTactile Local policy learning; removing tactile input reduces success rates by average 40%.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.13762) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **51 Imitation Learning with Tactile Feedback** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Touch begins where vision ends」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **51 Imitation Learning with Tactile Feedback**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_touch_2506_13762_touch-begins-where-vision-ends-generaliz.md`](../../sources/papers/sun_awesome_touch_2506_13762_touch-begins-where-vision-ends-generaliz.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2506.13762>
- [arXiv PDF](https://arxiv.org/pdf/2506.13762)：已通读方法部分（实验与附录以略读为准）：§2 相关工作、§3.1–3.3 方法；§4 实验仅略读。

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2506.13762)
