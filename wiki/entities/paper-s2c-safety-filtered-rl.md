---
type: entity
tags: [paper, safe-rl, reinforcement-learning, game]
status: complete
updated: 2026-09-27
arxiv: "2609.27312"
related:
  - ../overview/embodied-research-12-papers-recover-wam-technology-map.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/s2c_safety_filtered_rl_arxiv_2609_27312.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md
summary: "S2C（2609.27312）：先学鲁棒安全过滤器，再在过滤环境中训练竞争策略；达阵游戏测胜率与可利用性。"
---

# S2C

**Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning**（[arXiv:2609.27312](https://arxiv.org/abs/2609.27312)，[项目页](https://alliance-ai.cs.jhu.edu/s2c/)）收录自 [具身智能小站 12 篇盘点（恢复/WAM 专题）](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)。

## 一句话定义

**Safety-filtered RL 把安全约束放进训练环，使策略在竞争任务中既强又更难被 exploit。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SR | Success Rate | 任务成功率 |
| WAM | World Action Model | 联合预测未来观测与动作 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- 纳入 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 与同期失败恢复、异步 WAM、接触感知、持续学习、安全 RL 条目横向对照。
- 步骤 2.5 开源结论：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.27312](https://arxiv.org/abs/2609.27312) |
| **项目页** | https://alliance-ai.cs.jhu.edu/s2c/ |
| **代码** | — |
| **开源** | **待发布** |

## 实验与评测（公众号口径）

- 指标与数字以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md) 与 **原文 PDF** 为准；读复现前核对仿真/真机与 attempt 定义。


## 源码运行时序图

**不适用**（无统一官方入口或未开源）。

## 结论

**总判：S2C 适合作为「Safety-filtered RL 把安全约束放进训练环，使策略在竞争任务中既强又更难被 exploit。…」方向的入口页；机制细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 对照，避免与其它 arXiv 混淆。
2. 开源为 **待发布** 时从项目页/GitHub 再核实一次再写复现计划。
3. 涉及异步 WAM 或恢复评测时，同时记录 **正常起点 SR** 与 **偏差后恢复率**（若适用）。

## 关联页面

- [具身研究 12 篇（恢复/WAM）技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/s2c_safety_filtered_rl_arxiv_2609_27312.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)

## 推荐继续阅读

- [arXiv:2609.27312](https://arxiv.org/abs/2609.27312)
- [项目页](https://alliance-ai.cs.jhu.edu/s2c/)

