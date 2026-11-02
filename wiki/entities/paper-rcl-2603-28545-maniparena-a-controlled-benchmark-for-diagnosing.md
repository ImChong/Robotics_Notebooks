---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2603.28545"
venue: "2026"
summary: "ManipArena evaluates manipulation policies through controlled physical robot trials, task schemas and subgoal scoring. Its strongest lesson is that training recipes and model provenance affect rankings alongside architec"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2603_28545_maniparena-a-controlled-benchmark-for-di.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# ManipArena

**ManipArena: A Controlled Benchmark for Diagnosing Generalization in Real-Robot Manipulation** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 036/564** 篇，分组 **Benchmarks & simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

ManipArena evaluates manipulation policies through controlled physical robot trials, task schemas and subgoal scoring. Its strongest lesson is that training recipes and model provenance affect rankings alongside architecture. Language grounding and demonstration selection produce substantial reported gains, but smal...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- ManipArena evaluates manipulation policies through controlled physical robot trials, task schemas and subgoal scoring. Its strongest lesson is that training recipes and model provenance affect rankings alongside architecture. Language grounding and demonstration selection produce substantial reported gains, but smal...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    SCH["任务 schema：任务阶段、物体/布局变量、<br/>干扰物条件、留出条件"] --> TR["训练集采样 c ~ P_train<br/>20 个真机任务 / 10,812 条示范"]
    SCH --> EV["同一 schema 诱导三类评测分布"]
    TR --> POL["待测策略训练<br/>（语言标注 L1 / L2 / L3 可控）"]
    EV --> B1["第 1–4 次试验：域内配置 P_ID"]
    EV --> B2["第 5–8 次试验：受控视觉/配置偏移 P_shift"]
    EV --> B3["第 9–10 次试验：最强留出条件 P_held"]
    POL --> RUN["真机 10 次试验协议<br/>绿幕、固定光照与相机"]
    B1 --> RUN
    B2 --> RUN
    B3 --> RUN
    RUN --> SC["子目标部分得分<br/>每次试验 10 分，按 w_k 加权"]
    SC --> AGG["汇总：得分 S 与成功率 SR（成功需 ≥ 9/10 分）"]
    SIM["配对仿真（3 个桌面任务）<br/>仅作真机验证前的筛查诊断"] -.-> POL
```

图依据：论文 §3.1–3.5（评测协议；本图为基准的评测流程，非模型结构）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 036/564 |
| 分组 | Benchmarks & simulators |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2603.28545> |
| 子类 / 象限 | 机器人操作基准 · 泛化评测 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

ManipArena evaluates manipulation policies through controlled physical robot trials, task schemas and subgoal scoring. Its strongest lesson is that training recipes and model provenance affect rankings alongside architecture. Language grounding and demonstration selection produce substantial reported gains, but smal...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.28545) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Benchmarks & simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ManipArena」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Benchmarks & simulators**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2603_28545_maniparena-a-controlled-benchmark-for-di.md`](../../sources/papers/rcl_awesome_wam_2603_28545_maniparena-a-controlled-benchmark-for-di.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2603.28545>
- [arXiv PDF](https://arxiv.org/pdf/2603.28545)：已通读方法部分（实验与附录以略读为准）：§1–3（ManipArena Benchmark：schema 任务设计、分层真机评测、部分得分、语言标注与仿真扩展）；§4 实验与附录仅略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2603.28545)
