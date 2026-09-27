---
type: entity
tags: [paper, benchmark, manipulation, vla, evaluation, recovery]
status: complete
updated: 2026-09-27
arxiv: "2609.28952"
code: https://github.com/RUCKBReasoning/RoboRecover
related:
  - ../overview/embodied-research-12-papers-recover-wam-technology-map.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/roborecover_arxiv_2609_28952.md
  - ../../sources/repos/roborecover.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md
summary: "RoboRecover（2609.28952）：从执行偏差后的中间状态评测策略恢复；LIBERO 200 场景下正常起点与恢复排名不一致。"
---

# RoboRecover

**RoboRecover: Benchmarking Robot Policy Recovery under Execution Deviations**（[arXiv:2609.28952](https://arxiv.org/abs/2609.28952)）收录自 [具身智能小站 12 篇盘点（恢复/WAM 专题）](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)。

## 一句话定义

**动作前缀回放固定偏差起点，测策略能否修复关系并完成原任务——把「从头成功」与「出错后继续」拆成两个维度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SR | Success Rate | 任务成功率 |
| WAM | World Action Model | 联合预测未来观测与动作 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- 纳入 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 与同期失败恢复、异步 WAM、接触感知、持续学习、安全 RL 条目横向对照。
- 步骤 2.5 开源结论：**已开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28952](https://arxiv.org/abs/2609.28952) |
| **项目页** | — |
| **代码** | https://github.com/RUCKBReasoning/RoboRecover |
| **开源** | **已开源** |

## 实验与评测（公众号口径）

- 指标与数字以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md) 与 **原文 PDF** 为准；读复现前核对仿真/真机与 attempt 定义。
- RoboRecover：UniFOLM 正常 98.83% vs 恢复 48.00%；π₀.₅ 94.17% vs 64.40%（LIBERO 固定 200 场景，公众号口径）。

## 源码运行时序图

**不适用**（请按 GitHub README 入口自行补 sequenceDiagram；入库日未逐仓核对）。

## 结论

**总判：RoboRecover 适合作为「动作前缀回放固定偏差起点，测策略能否修复关系并完成原任务——把「从头成功」与「出错后继续」拆成两个维度。…」方向的入口页；机制细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 对照，避免与其它 arXiv 混淆。
2. 开源为 **已开源** 时从项目页/GitHub 再核实一次再写复现计划。
3. 涉及异步 WAM 或恢复评测时，同时记录 **正常起点 SR** 与 **偏差后恢复率**（若适用）。

## 关联页面

- [具身研究 12 篇（恢复/WAM）技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/roborecover_arxiv_2609_28952.md)
- [RoboRecover 仓库归档](../../sources/repos/roborecover.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)

## 推荐继续阅读

- [arXiv:2609.28952](https://arxiv.org/abs/2609.28952)
- [GitHub 仓库](https://github.com/RUCKBReasoning/RoboRecover)

