---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2506.06361"
venue: "arXiv 2025"
summary: "Open-source, Gymnasium-compatible benchmark for active tactile perception tasks including localization, classification, and volume."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2506_06361_tactile-mnist-benchmarking-active-tactil.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Tactile MNIST

**Tactile MNIST: Benchmarking Active Tactile Perception** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 053/065** 篇，分组 **Tactile Simulation & Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Open-source, Gymnasium-compatible benchmark for active tactile perception tasks including localization, classification, and volume.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Open-source, Gymnasium-compatible benchmark for active tactile perception tasks including localization, classification, and volume.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    FORM["主动感知形式化 §3.1：POMDP 特例；隐状态 S = S_base × Y*，动作 A = A_base × Y，预测不影响转移；奖励 R = R_base - ℓ(y*, y)，每步都要做预测"]
    FORM --> APG["Active Perception Gym（ap_gym，§3.2）：Gymnasium 兼容，每个任务带可微损失 ℓ 与预测目标 y*，自动计算准确率或欧氏距离等指标"]
    APG --> TMBS["Tactile MNIST Benchmark Suite（TMBS，§3.3）：智能体控制模拟 GelSight Mini，仅凭触觉与本体感觉"]
    TMBS --> ENVS["四个环境：TactileMNIST（分类）/ TactileMNISTVolume（体积估计）/ Toolbox（工具位姿估计）/ Starstruck（数星星计数）"]
    TMBS --> REND["触觉渲染三种模式：Depth / Taxim / CycleGAN（目前仅 TactileMNIST 环境）"]
    MNIST3D["MNIST 3D（§3.4）：由 500×500 高分辨率 MNIST 变体自动生成的 13,580 个可 3D 打印网格"] --> PRINT["3D 打印 600 个数字（§3.5）"]
    PRINT --> REAL["Franka Research 3 + GelSight Mini：在网格内随机位置按压，法向力超过 5 N 即停止并记录时间戳；每个数字 256 次，共 153,600 次触碰；按 90% / 10% 划分训练与测试"]
    REAL --> CGAN["训练 CycleGAN：深度图到触觉图的风格迁移，用于逼真渲染与 sim-to-real"]
    CGAN --> REND
    ENVS --> PROTO["评测协议 §3.6：exploration efficiency（平均预测准确率 / 最终预测准确率）与 sample efficiency"]
    PROTO --> BASE["基线（Fig. 5）：TAP-SAC / TAP-CrossQ / TAP-PPO 等"]
```

图依据：基准构建与评测协议图，含数据采集流程（POMDP 形式化 → ap_gym / TMBS 环境 → MNIST 3D 与真实触觉数据采集 → CycleGAN 渲染 → 评测协议）：论文 §3.1–§3.6、Fig. 1–4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 053/065 |
| 分组 | Tactile Simulation & Sim2Real |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.06361> |

## 核心机制（归纳）

### 策展导读要点

Open-source, Gymnasium-compatible benchmark for active tactile perception tasks including localization, classification, and volume.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.06361) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile Simulation & Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Tactile MNIST」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_touch_2506_06361_tactile-mnist-benchmarking-active-tactil.md`](../../sources/papers/sun_awesome_touch_2506_06361_tactile-mnist-benchmarking-active-tactil.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2506.06361>
- [arXiv PDF](https://arxiv.org/pdf/2506.06361)：已通读方法部分（实验与附录以略读为准）：摘要、§1–§3（含 Fig. 1–4、Table 1–2）、§4 基线名称与 Fig. 5 图注；附录 C–F 与基线结果数字以略读为准

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2506.06361)
