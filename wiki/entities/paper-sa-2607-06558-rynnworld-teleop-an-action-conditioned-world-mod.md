---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.06558"
venue: "arXiv 2026"
code: https://github.com/alibaba-damo-academy/RynnWorld-Teleop
summary: "Action-conditioned world model for digital teleoperation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_06558_rynnworld-teleop-an-action-conditioned-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
  - ../../sources/repos/rynnworld-teleop.md
---

# RynnWorld-Teleop

**RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 332/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Action-conditioned world model for digital teleoperation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Action-conditioned world model for digital teleoperation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TR["训练"]
        P1["阶段一：第一人称人类视频预训练<br/>EgoDex + VITRA，配对 (2D 骨架, RGB)"] --> P2["阶段二：机器人域适配<br/>配对遥操作数据，人手姿态经 IK 映射到机器人动作"]
        P2 --> T["双向教师：Wan-I2V 视频 DiT<br/>+ 并行 control patch-embedding（分布对齐、可学习门控 α）"]
        T --> D1["因果 flow-matching 预热<br/>得到帧级因果学生（因果 mask + KV cache，参考图作 sink token）"]
        D1 --> D2["DMD 分布匹配蒸馏<br/>4 步采样，梯度跨 chunk 经 KV cache 回传"]
    end
    subgraph INF["数字遥操作数据生成"]
        H["操作员手部姿态流（21 关节）"] --> SK["渲染深度感知骨架视频 S（16 FPS）"]
        SK --> VAE["预训练 VAE 编码 → 控制潜变量 c"]
        REF["参考图 I_ref"] --> GEN
        VAE --> GEN["蒸馏后的因果学生<br/>按 81 帧 chunk 生成机器人第一人称视频"]
        GEN --> VID["生成视频"]
        GEN -- "后续 chunk 以机器人相机真实帧作新 I_ref（re-anchoring，抑制漂移）" --> GEN
        H --> RT["Retargeting：坐标变换链 + DLS 逆运动学<br/>得到 54 维机器人动作 a"]
        VID --> PAIR["成对 (视频, 动作) 轨迹 → 模仿学习"]
        RT --> PAIR
    end
    D2 --> GEN
```

图依据：论文 §3.1–3.5 / Fig.3、§4.1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 332/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.06558> |
| 代码/项目 | <https://github.com/alibaba-damo-academy/RynnWorld-Teleop> |

## 核心机制（归纳）

### 策展导读要点

Action-conditioned world model for digital teleoperation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.06558) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RynnWorld-Teleop」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [RynnWorld-Teleop 源码归档](../../sources/repos/rynnworld-teleop.md)（<https://github.com/alibaba-damo-academy/RynnWorld-Teleop>）

- [`sources/papers/sun_awesome_wm_2607_06558_rynnworld-teleop-an-action-conditioned-w.md`](../../sources/papers/sun_awesome_wm_2607_06558_rynnworld-teleop-an-action-conditioned-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.06558>
- [arXiv PDF](https://arxiv.org/pdf/2607.06558)：已通读方法部分（实验与附录以略读为准）：摘要、§3 方法（3.1–3.5）、§4 数字遥操作系统（4.1–4.2）、Fig.3

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.06558)
