---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2606.13877"
venue: "arXiv 2026"
summary: "Benchmark and systematic empirical study of vision-tactile world models across 12 contact-rich tasks; finds spatially structured and temporally continuous representations achieve strongest planning performance."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2606_13877_contactworld-what-matters-in-vision-tact.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# ContactWorld

**ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 060/065** 篇，分组 **Visuo-Tactile World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Benchmark and systematic empirical study of vision-tactile world models across 12 contact-rich tasks; finds spatially structured and temporally continuous representations achieve strongest planning performance.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Benchmark and systematic empirical study of vision-tactile world models across 12 contact-rich tasks; finds spatially structured and temporally continuous representations achieve strongest planning performance.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    O["多模态观测：腕部/前视 RGB、点云（1024 点）、TacRGB、TacDepth、TacFF"] --> EN["模态专用编码器：IMPALA 式 CNN / PointNet 式空间编码器 / TacFF 逐 taxel 编码"]
    EN --> LF["Latent Fusion：各模态隐变量融合"]
    LF --> WM["动作条件隐空间世界模型：自回归预测未来隐变量（JEPA 式，不重建观测）"]
    A["动作嵌入"] --> WM
    WM --> L["训练目标：预测损失 + VICReg 正则 + 时间相似 + 逆动力学（IDM）"]
    LF --> P1["规划：编码当前观测 z_t 与目标观测 z_g"]
    P1 --> CEM["CEM 采样候选动作序列（100 条，4 次迭代，保留 8 个精英）"]
    CEM --> RO["世界模型隐空间自回归 rollout（6 步）"]
    RO --> C["终端隐空间代价 ‖ẑ_(t+H) − z_g‖²"]
    C --> CEM
    C --> EX["执行首个动作并重规划（receding-horizon MPC，10 Hz）"]
    EX --> O
```

图依据：论文 §III-B / §III-C / Fig.3（训练与规划流程）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 060/065 |
| 分组 | Visuo-Tactile World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.13877> |

## 核心机制（归纳）

### 策展导读要点

Benchmark and systematic empirical study of vision-tactile world models across 12 contact-rich tasks; finds spatially structured and temporally continuous representations achieve strongest planning performance.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.13877) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Visuo-Tactile World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ContactWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Visuo-Tactile World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2606_13877_contactworld-what-matters-in-vision-tact.md`](../../sources/papers/sun_awesome_touch_2606_13877_contactworld-what-matters-in-vision-tact.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2606.13877>
- [arXiv PDF](https://arxiv.org/pdf/2606.13877)：已通读方法部分（实验与附录以略读为准）：§III ContactWorld（任务、模态表示、世界模型学习与规划评测）、§I 引言；§IV 起实验仅略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2606.13877)
