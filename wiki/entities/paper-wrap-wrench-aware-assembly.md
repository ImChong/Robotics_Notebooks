---
type: entity
tags: [paper, manipulation, multi-robot, planning, assembly]
status: complete
updated: 2026-09-26
arxiv: "2609.29407"
related:
  - ../overview/embodied-research-12-papers-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/wrap-wrench-aware-assembly_arxiv_2609_29407.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md
summary: "Wrap（arXiv:2609.29407）：无夹具多机装配；联动 wrench、抓取、交接与运动规划；需零件几何与装配依赖。"
---

# Wrap

**Wrap**（*Fixtureless Wrench-aware Multi-Robot Assembly Planning*，[arXiv:2609.29407](https://arxiv.org/abs/2609.29407)，[项目页](https://www.vhartmann.com/wrap)）收录自 [具身智能小站 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)。

## 一句话定义

**多机器人协同托稳并装配时，把受力、抓取与运动规划放在同一 wrench-aware 搜索里，避免只靠专用夹具。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| IL | Imitation Learning | 模仿学习 |
| SR | Success Rate | 任务成功率 |
| WM | World Model | 世界模型 |

## 为什么重要

- 纳入 [12 篇具身研究清单](../../wiki/overview/embodied-research-12-papers-technology-map.md) 主线，与同期 VLA / 接触 / 规划 / 安全论文可横向对照。
- 公众号强调的可操作读法：先看 **任务信息需求**（如 PolyUMI 旋灯泡仍以视觉最优）与 **评测口径**（如 Self-Adaptive 多 trial、BeyondRetarget 仿真片段非真机 SR）。
- 开源状态（步骤 2.5）：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.29407](https://arxiv.org/abs/2609.29407) |
| **项目页** | https://www.vhartmann.com/wrap |
| **代码** | 截至入库日未列 |
| **开源** | **待发布** |

## 实验与评测（公众号口径）

- 指标与消融以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md) 与 **原文 PDF** 为准；本页不复制整表。
- 读复现前先核对：样本规模、是否仿真/真机、是否允许多次 attempt。

## 源码运行时序图

**不适用**（截至 2026-09-26 项目页未提供可运行官方代码仓库；见 [sources/papers/wrap-wrench-aware-assembly_arxiv_2609_29407.md](../../sources/papers/wrap-wrench-aware-assembly_arxiv_2609_29407.md)）。

## 结论

**总判：Wrap 适合作为「多机器人协同托稳并装配时，把受力、抓取与运动规划放在同一 wrench-awar…」方向的入口页；细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md) 对照选型，避免与同名不同 arXiv 的工作混淆（如 RAPID vs RAPID-VLM-RL）。
2. 开源为 **待发布** 时优先从项目页 Code 区核实，再写复现计划。
3. 长程 / 部署类条目（AdaHVLA、HarnessPAI、Self-Adaptive VLA）同时记录 **成功率定义** 与 **失败恢复预算**。

## 关联页面

- [具身研究 12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md)
- [Manipulation](../tasks/manipulation.md)
- [VLA](../methods/vla.md)

## 参考来源

- [wrap-wrench-aware-assembly 论文归档](../../sources/papers/wrap-wrench-aware-assembly_arxiv_2609_29407.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)

## 推荐继续阅读

- [arXiv:2609.29407](https://arxiv.org/abs/2609.29407)
- [项目页](https://www.vhartmann.com/wrap)
