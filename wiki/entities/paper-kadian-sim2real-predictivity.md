---
type: entity
tags:
  - paper
  - sim2real
  - evaluation
  - benchmark
status: complete
updated: 2026-09-20
venue: "RA-L 2020"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/freedof_sim2real_38_kadian-sim2real-predictivity.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "系统研究仿真评测指标能否预测真机表现，为 benchmark 设计提供依据。"
---

# Sim2Real predictivity: does evaluation in simulation predict real-world performance?

**Sim2Real predictivity: does evaluation in simulation predict real-world performance?**（RA-L 2020）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[38/44]**，归类 **监控与评测**。

## 一句话定义

系统研究仿真评测指标能否预测真机表现，为 benchmark 设计提供依据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真机 |
| RA-L | Robotics and Automation Letters | IEEE 机器人快报 |
| Benchmark | Benchmark | 标准化评测套件 |

## 为什么重要

- 文内评测侧代表；呼应「离线指标与硬件表现脱钩」。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **监控与评测** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | RA-L 2020 |
| **文内章节** | 监控与评测 |
| **要点** | 跨 sim/real 任务对比相关性；提出 predictivity 概念。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 实验与评测

- **本页为索引级节点**（FreeDof 44 篇梳理 [38/44]）：正文固化文内角色与机制要点，**未转存原文实验表**。
- **回原文须核对的证据**：本页要点是「跨 sim/real 任务对比相关性；提出 predictivity 概念」，对应证据是同一批策略在 sim 与 real 上成绩的**相关性统计**，而非单个策略的成功率。
- **读法：** 先对齐平台、任务、指标定义与成功阈值，再读任何数字；勿从公众号摘录外推。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **文内路线** | 归类 **监控与评测**；同路线其他节点见 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) |
| **横比口径** | 相关性只对「某个仿真器 + 某个任务族」成立；换仿真器或换任务族必须重测，不能假设predictivity 可继承。 |
| **开源状态** | **待核实** — 部署 / 复现前以项目页或原文 Code availability 为准 |

## 结论

**仿真 leaderboard 高不等于部署可用；需读 predictivity 文献校准期望。**

1. 文内角色：监控与评测 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：跨 sim/real 任务对比相关性；提出 predictivity 概念。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)
- [具身大模型评测基准选型闭环知识链](../queries/embodied-eval-benchmark-selection-loop.md) — 本文系统追问「仿真评测指标能否预测真机表现」，是该闭环第 ④ 层「sim↔real 评测 gap 校准：评测结论能否外推真机」的奠基性参考

## 参考来源

- [freedof_sim2real_38_kadian-sim2real-predictivity.md](../../sources/papers/freedof_sim2real_38_kadian-sim2real-predictivity.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
