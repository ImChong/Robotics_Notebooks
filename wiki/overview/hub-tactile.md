---
type: overview
tags: [hub, hub-tactile, haptic, force, contact, visuo-tactile]
status: complete
updated: 2026-07-31
summary: "触觉与力觉闭环知识链汇总：覆盖触觉传感、视触觉融合、阻抗/力控与接触估计，强调「摸得着」对抓取与 loco-manip 稳定性的作用。"
---

# 触觉与力觉（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=contact-manipulation) 与 [路线页](../../roadmap/depth-contact-manipulation.md)。

## 一句话定义

**触觉知识链** 研究机器人如何通过 **力、触觉与接触状态** 闭环调节交互，使抓取、装配与 loco-manip 在不确定接触下仍稳定可控。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Tactile | Tactile Sensing | 表面/接触力学感知 |
| Haptic | Haptic Feedback | 力反馈与遥操作触觉 |
| Impedance | Impedance Control | 力-位关系调节的交互控制 |
| Visuo-Tactile | Visuo-Tactile Fusion | 视觉与触觉联合表征 |
| Wrench | Wrench (Force/Torque) | 六维力/力矩测量 |

## 为什么重要

- **视觉有盲区**：遮挡、反光、透明物体下，触觉是接触真相。
- **硬位置控制在接触瞬间易崩**：需要阻抗/力控与接触估计。
- **V21 知识链主线**：从 GelSight 类传感器到全身 loco-manip 的力闭环。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 传感 | 触觉模态与硬件 | [Tactile Sensing](../concepts/tactile-sensing.md) |
| 融合 | 视+触联合 | [Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md) |
| 控制 | 阻抗与力控基础 | [Impedance Control](../concepts/impedance-control.md)、[Force Control Basics](../concepts/force-control-basics.md) |
| 估计 | 接触状态/力 | [Contact Estimation](../concepts/contact-estimation.md) |
| 执行 | 力位混合 | [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md) |
| 操作员渲染 | 可穿戴力触觉显示 | [HapMorph](../entities/paper-hapmorph-pneumatic-haptic-render.md)（气动 AFPA 解耦尺寸+刚度） |
| 抓取精修 | 纯触觉目标条件伺服 | [TacRefineNet](../entities/paper-tacrefinenet-tactile-grasp-refinement.md)（Siamese · 外在灵巧 regrasp · arXiv:2509.25746） |
| 传感器选型基准 | 跨模态真机 IL 对比 | [TacO](../entities/paper-taco-tactile-sensor-benchmark.md)（六传感器 × 三任务 ACT；无通用最佳模态 · arXiv:2605.21976） |
| 可变形接触安全评测 | Goal vs Safety Success | [SoftVTBench](../entities/paper-softvtbench.md)（Isaac Sim FEM + GelSight；π₀.₅ VO/VT · arXiv:2607.04234） |
| 数据+策略栈 | 力场表征 / VTLA / 触觉 WAM | [NeoteAI 𝒩₀](../entities/neoteai.md)（OpenNeoData 5k h；Foundation / VTLA / TWAM） |

## 与其他知识链的关系

- **[抓取](./hub-grasp.md)**：稳定抓取依赖力闭环。
- **[WBC](./hub-wbc.md)**：全身力分配与接触约束。
- **[通信协议](./hub-communication.md)**：EtherCAT 等低延迟总线服务力控环路。

## 关联页面

- [HapMorph（论文实体）](../entities/paper-hapmorph-pneumatic-haptic-render.md) — VR/遥操作操作员侧可穿戴气动触觉
- [TacRefineNet（论文实体）](../entities/paper-tacrefinenet-tactile-grasp-refinement.md) — 边缘突出物体的纯触觉抓取精修
- [TacO（触觉传感器操作基准）](../entities/paper-taco-tactile-sensor-benchmark.md) — 跨模态触觉硬件 × 真机 ACT 选型证据
- [SoftVTBench（可变形视触觉安全基准）](../entities/paper-softvtbench.md) — Goal/Safety Success；触觉抬高软体安全率
- [NeoteAI / 𝒩₀ 三件套](../entities/neoteai.md) — OpenNeoData + NeoForce + VTLA/TWAM
- [Teleoperation](../tasks/teleoperation.md) — 操作员力反馈与示范采集
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md)（接触安全）

## 参考来源

- 本库归纳自 [Tactile Sensing](../concepts/tactile-sensing.md)、[Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md)、[Impedance Control](../concepts/impedance-control.md)
- **ingest 档案：** [sources/papers/hapmorph_arxiv_2509_05433.md](../../sources/papers/hapmorph_arxiv_2509_05433.md) — HapMorph 可穿戴气动多维触觉渲染（arXiv:2509.05433）
- **ingest 档案：** [sources/papers/tacrefinenet_arxiv_2509_25746.md](../../sources/papers/tacrefinenet_arxiv_2509_25746.md) — TacRefineNet 多指触觉抓取精修（arXiv:2509.25746）
- **ingest 档案：** [sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md](../../sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md) — TacO 触觉传感器真机 IL 基准（arXiv:2605.21976）
- **ingest 档案：** [sources/papers/softvtbench_arxiv_2607_04234.md](../../sources/papers/softvtbench_arxiv_2607_04234.md) — SoftVTBench 可变形视触觉安全基准（arXiv:2607.04234）
- **ingest 档案：** [sources/papers/n0_foundation.md](../../sources/papers/n0_foundation.md) — 𝒩₀-Foundation / OpenNeoData
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`tactile` 命中规则）
