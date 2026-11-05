---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2603.25135"
summary: "策展清单收录条目；细节以原文 PDF / 项目页为准。"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2603_25135_egoxtreme.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoXtreme

**EgoXtreme** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 231/249** 篇，分组 **Object & HOI Sub-collections**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

策展清单收录条目；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- 策展清单收录条目；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["Aria glasses 采集 egocentric RGB<br/>15 名参与者，775.5 分钟，30fps"] --> B["三类场景：Industrial Maintenance / Emergency Rescue / Sports<br/>8 种光照类型，部分加入烟雾"]
    B --> C["GT 位姿：OptiTrack 动捕（头显反光标记 + 物体刚体标记）"]
    C --> D["Umeyama 对齐 SLAM 与动捕轨迹 → Kalman 滤波修正漂移<br/>→ 手工时间偏移校正与插值"]
    D --> E["划分：train 518.8 / val 80.7 / test 176 分钟"]
    E --> F["§4.1 零样本基线：FoundPose / GigaPose / PicoPose<br/>指标 ADD(-S) recall、MSSD、MSPD"]
    E --> G["§4.2 先做去模糊 / 去雾 / 低光增强，再跑位姿估计器"]
    E --> H["§4.3 体育场景的位姿跟踪：三种时序策略对比逐帧基线"]
```

图依据：数据集采集与评测流程图，论文 §3.1–§3.3（Data collection and processing、Dataset configuration，Fig. 4）与 §4.1–§4.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 231/249 |
| 分组 | Object & HOI Sub-collections |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2603.25135> |

## 核心机制（归纳）

### 策展导读要点

策展清单收录条目；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.25135) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Object & HOI Sub-collections** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoXtreme」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Object & HOI Sub-collections**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2603_25135_egoxtreme.md`](../../sources/papers/sun_awesome_ego_2603_25135_egoxtreme.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2603.25135>
- [arXiv PDF](https://arxiv.org/pdf/2603.25135)：已通读摘要、Introduction、§3 数据集（场景、采集与标注处理、数据配置）与 §4 实验设置段落；结果表与附录未逐项核对。

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2603.25135)
