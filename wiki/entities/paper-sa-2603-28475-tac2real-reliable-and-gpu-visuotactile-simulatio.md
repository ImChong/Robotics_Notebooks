---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2603.28475"
venue: "arXiv 2026"
summary: "GPU visuotactile simulation for online RL and zero-shot real-world deployment."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2603_28475_tac2real-reliable-and-gpu-visuotactile-s.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Tac2Real

**Tac2Real: Reliable and GPU Visuotactile Simulation for Online Reinforcement Learning and Zero-Shot Real-World Deployment** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 050/065** 篇，分组 **Tactile Simulation & Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

GPU visuotactile simulation for online RL and zero-shot real-world deployment.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- GPU visuotactile simulation for online RL and zero-shot real-world deployment.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["RL 智能体发出动作"] --> P["物理引擎执行一步<br/>Isaac Lab / Isaac Gym / PyBullet / MuJoCo 等"]
    P --> R["提取被抓物体与触觉传感器的<br/>相对位姿及线/角速度"]
    R --> T["外置 PNCG-IPC 触觉仿真<br/>Ray 集群，每 GPU 管理多个环境"]
    T --> K["GelSight Mini 标记点位移场 9×7<br/>（k 近邻映射 + 加权插值）"]
    K --> O["与引擎基础观测拼成组合观测"]
    P --> O
    O --> A
    subgraph TacAlign["TacAlign 缩小 sim2real 差距"]
        C1["机器人控制对齐<br/>交替最小化轨迹差异，校准 kp"]
        C2["基础 IPC 标定<br/>CMA-ES 拟合 E、ν、ρ、μ"]
        C3["任务级标定<br/>微调 Isaac 的 μ 与接触刚度"]
        C4["域随机化<br/>控制增益、摩擦、位姿、传感噪声"]
    end
    C1 -.-> P
    C2 -.-> T
    C3 -.-> P
    C4 -.-> A
```

图依据：论文 §3.1–3.3、§4.1–4.4 / Fig.3、Fig.4（TacAlign 各项作用于哪个环节的虚线为按所标定参数的归属推断）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 050/065 |
| 分组 | Tactile Simulation & Sim2Real |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.28475> |

## 核心机制（归纳）

### 策展导读要点

GPU visuotactile simulation for online RL and zero-shot real-world deployment.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.28475) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile Simulation & Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Tac2Real」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tactile Simulation & Sim2Real**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2603_28475_tac2real-reliable-and-gpu-visuotactile-s.md`](../../sources/papers/sun_awesome_touch_2603_28475_tac2real-reliable-and-gpu-visuotactile-s.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2603.28475>
- [arXiv PDF](https://arxiv.org/pdf/2603.28475)：已通读方法部分（实验与附录以略读为准）：§1–4（PNCG-IPC、触觉表示、物理引擎集成、TacAlign 四部分）；§5 实验仅略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2603.28475)
