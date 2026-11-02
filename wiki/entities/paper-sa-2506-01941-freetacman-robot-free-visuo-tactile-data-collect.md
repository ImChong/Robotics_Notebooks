---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-16
arxiv: "2506.01941"
venue: "arXiv 2025"
summary: "Human-centric, robot-free data collection system leveraging dexterity and force feedback of human motion; comprises 10k demonstration trajectories across 50 tasks."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ./paper-fomi.md
sources:
  - ../../sources/papers/sun_awesome_touch_2506_01941_freetacman-robot-free-visuo-tactile-data.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# FreeTacMan

**FreeTacMan: Robot-free Visuo-Tactile Data Collection System for Contact-rich Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 041/065** 篇，分组 **Tactile Sensors & Hardware**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Human-centric, robot-free data collection system leveraging dexterity and force feedback of human motion; comprises 10k demonstration trajectories across 50 tasks.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Human-centric, robot-free data collection system leveraging dexterity and force feedback of human motion; comprises 10k demonstration trajectories across 50 tasks.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["操作者指尖佩戴视触觉传感器的 in-situ 夹爪（采集接口）"] --> B["同步记录：腕部相机 RGB、两路视触觉图像、夹爪宽度"]
    A --> C["NOKOV 动捕 240 Hz 追踪末端 6D 位姿，下采样与图像对齐"]
    B --> D["每帧同步为 30 Hz 的与本体无关轨迹数据"]
    C --> D
    D --> E["IKPY 逆运动学：位姿映射为目标机器人关节位置（动作表示）"]
    D --> F["触觉预训练：视觉编码器 fv 冻结，触觉编码器 ft 微调；时空融合对比损失（主正样本时间对齐、次正样本相邻时刻、记忆库负样本）"]
    F --> G["预训练触觉编码器"]
    E --> H["视觉嵌入与触觉嵌入拼接，输入 ACT，预测 7-DoF 关节位置"]
    G --> H
    H --> I["同一模块化传感器与夹爪换到执行接口（Piper / Franka）部署策略"]
```

图依据：论文 §III.A–C / Fig. 2、Fig. 3、式 (1)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 041/065 |
| 分组 | Tactile Sensors & Hardware |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.01941> |

## 核心机制（归纳）

### 策展导读要点

Human-centric, robot-free data collection system leveraging dexterity and force feedback of human motion; comprises 10k demonstration trajectories across 50 tasks.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.01941) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile Sensors & Hardware** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「FreeTacMan」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tactile Sensors & Hardware**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)
- [FOMI](./paper-fomi.md) — 无机器人数据采集的力反馈路线：机械传力而非视触觉传感

## 参考来源

- [`sources/papers/sun_awesome_touch_2506_01941_freetacman-robot-free-visuo-tactile-data.md`](../../sources/papers/sun_awesome_touch_2506_01941_freetacman-robot-free-visuo-tactile-data.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2506.01941>
- [arXiv PDF](https://arxiv.org/pdf/2506.01941)：已通读方法部分（实验与附录以略读为准）：§III 方法（A 硬件设计、B 人到机器人数据迁移、C 触觉预训练与策略学习）、§IV.A 数据集

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2506.01941)
