---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2506.01944"
venue: "arXiv 2025"
summary: "Models human tactile-proprioceptive signals during manipulation; trains closed-loop imitation learning policy; achieves 77% success rate across 5 force-sensitive tasks."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2506_01944_feel-the-force-contact-driven-learning-f.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Feel the Force

**Feel the Force: Contact-Driven Learning from Humans (FTF)** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 022/065** 篇，分组 **51 Imitation Learning with Tactile Feedback**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Models human tactile-proprioceptive signals during manipulation; trains closed-loop imitation learning policy; achieves 77% success rate across 5 force-sensitive tasks.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Models human tactile-proprioceptive signals during manipulation; trains closed-loop imitation learning policy; achieves 77% success rate across 5 force-sensitive tasks.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["训练：仅人类演示"]
        A["两台标定 RealSense 相机 + 触觉手套（AnySkin 磁力计，取力向量范数）"] --> B["MediaPipe 手部关键点 → 双目三角化得 3D 手部关键点"]
        B --> C["拇指与食指指尖中点为机器人位置；刚体变换得姿态；指距小于 7cm 为闭合"]
        C --> D["转为 N 个机器人关键点 Pr + 夹爪状态 Rg + 力值 Rf"]
        E["单帧人工标注物体关键点 → DIFT 传播到各演示首帧 → Co-Tracker 跟踪并三角化得物体点 Po"] --> F
        D --> F["Transformer 策略（MLP 编码点轨迹历史，每条轨迹一个 token）"]
        F --> G["预测未来机器人点轨迹、夹爪状态、夹爪力（action chunking + 指数时间平均）"]
    end
    subgraph DEPLOY["部署：Franka Panda + 单指 AnySkin 触觉"]
        H["DIFT 定位首帧物体关键点，Co-Tracker 跟踪读取状态"] --> I["策略输出动作块 → 机器人位姿 + 夹爪状态 + 目标力 F̂"]
        I --> J["夹爪闭合时：PD 力控制调节夹爪开度直至测得力逼近 F̂"]
        J --> K["末端位置控制执行（6Hz）"]
        K --> H
    end
    G -.-> I
```

图依据：论文 §3.1 数据采集 / §3.2 Embodiment Agnostic Scene Representation / §3.3 Policy Learning / §3.4 Inference（Algorithm 1–2）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 022/065 |
| 分组 | 51 Imitation Learning with Tactile Feedback |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.01944> |

## 核心机制（归纳）

### 策展导读要点

Models human tactile-proprioceptive signals during manipulation; trains closed-loop imitation learning policy; achieves 77% success rate across 5 force-sensitive tasks.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.01944) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **51 Imitation Learning with Tactile Feedback** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Feel the Force」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_touch_2506_01944_feel-the-force-contact-driven-learning-f.md`](../../sources/papers/sun_awesome_touch_2506_01944_feel-the-force-contact-driven-learning-f.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2506.01944>
- [arXiv PDF](https://arxiv.org/pdf/2506.01944)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3.1–3.4 方法与 Algorithm 1/2、§4 实验设置与任务；结果表略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2506.01944)
