---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2508.20840"
venue: "arXiv 2025"
summary: "Towards scalable robotic learning with primitive embodied world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2508_20840_learning-primitive-embodied-world-models.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Learning Primitive Embodied World Models

**Learning Primitive Embodied World Models: Towards Scalable Robotic Learning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 307/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Towards scalable robotic learning with primitive embodied world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Towards scalable robotic learning with primitive embodied world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["语言指令 + 当前观测 O_t<br/>如 Pick up the yellow tape measure"] --> B["Stage I: VLM 规划器 (Qwen2.5-VL-7B)<br/>Reasoner → Planner(LoRA-P) 拆成原语序列 P1..PN"]
    B --> C["Grounder(LoRA-G)<br/>为每个原语给出夹爪起点/终点像素位置 → 起止点热力图"]
    C --> D["Verifier 检验计划可执行性<br/>通过 / 拒绝"]
    D --> E["Stage II: 原语具身世界模型 PEWM(视频扩散)<br/>条件: 当前帧 + 原语文本(CLIP) + 起止热力图"]
    E --> F["生成未来帧短视频 (约 ≤2 秒)"]
    F --> G["Stage III: Gen6D 零样本位姿估计<br/>提取末端 6-DoF 轨迹"]
    G --> H["结合度量深度提升到 3D, 变换到机器人基座坐标系"]
    H --> I["机器人执行"]
    I --> J["更新观测 O_t+1, 取最新帧作为下一轮输入"]
    J -. 闭环自回归, 逐原语重复 .-> E
    D -. 二值开合夹爪原语 .-> K["跳过视频生成 直接符号控制"]
```

图依据：论文 §4.1–4.2 / Fig. 4、Fig. 5，附录 Fig. 6、Fig. 8 与附录 C（原语规划与视频生成）。"夹爪原语跳过视频生成"来自附录 C；Verifier 与 Grounder 的先后次序按 Fig. 5/8 推断，可能与原图略有出入。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 307/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2508.20840> |
| 项目页 | <https://qiaosun22.github.io/PrimitiveWorld/> |

## 核心机制（归纳）

### 策展导读要点

Towards scalable robotic learning with primitive embodied world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2508.20840) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Learning Primitive Embodied World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2508_20840_learning-primitive-embodied-world-models.md`](../../sources/papers/sun_awesome_wm_2508_20840_learning-primitive-embodied-world-models.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2508.20840>
- [arXiv PDF](https://arxiv.org/pdf/2508.20840)：已通读方法部分（实验与附录以略读为准）：§1–§2 原语定义、§4.1–4.2 应用与闭环流程、附录 C 方法细节；实验与数据集略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2508.20840)
