---
type: entity
tags: [paper, wam, manipulation, asynchronous-inference, sjtu]
status: complete
updated: 2026-09-27
arxiv: "2609.28927"
code: https://github.com/SJTU-DENG-Lab/Streaming-WAM
related:
  - ../overview/embodied-research-12-papers-recover-wam-technology-map.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/streaming_wam_arxiv_2609_28927.md
  - ../../sources/repos/streaming-wam.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md
summary: "Streaming-WAM（2609.28927）：已排定动作前缀条件化世界预测，异步推理与运动重叠；真机盖章任务回合耗时约 90s→38s（论文口径）。"
---

# Streaming-WAM

**Streaming-WAM: Action-Conditioned World–Action Model for Asynchronous Robot Manipulation**（[arXiv:2609.28927](https://arxiv.org/abs/2609.28927)）收录自 [具身智能小站 12 篇盘点（恢复/WAM 专题）](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)。

## 一句话定义

**World–Action 模型在推理延迟下用 action-conditioned 预测对齐即将发生的场景，使控制与 WAM 推理并行。**

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
| **arXiv** | [2609.28927](https://arxiv.org/abs/2609.28927) |
| **项目页** | — |
| **代码** | https://github.com/SJTU-DENG-Lab/Streaming-WAM |
| **开源** | **已开源** |

## 实验与评测（公众号口径）

- 指标与数字以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md) 与 **原文 PDF** 为准；读复现前核对仿真/真机与 attempt 定义。
- Streaming-WAM：真机盖章任务平均回合 ~90s→~38s；30 次评估、RTX 5090、30Hz 执行（论文口径）。

## 源码运行时序图

**不适用**（请按 GitHub README 入口自行补 sequenceDiagram；入库日未逐仓核对）。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [WAM 实时异步部署](./paper-wam-realtime-async.md) | 六种异步部署策略的对照实证，结论是训练时注入 **已承诺前缀**（train）综合最好、但无实验代码；Streaming-WAM 把 **已排定动作前缀** 作为世界预测条件，是这条路线的具体模型并 **已开源** |
| [GlanceWAM](./paper-glancewam.md) | 异步 **稀疏单帧前瞻**，把视频生成移出控制关键路径；Streaming-WAM 让 WAM 推理与 **运动执行重叠**，用 action-conditioned 预测对齐推理完成时的场景 |
| [LiMA](./paper-lima-async-dual-system-wam.md) · [DualWAM](./paper-dualwam.md) | 用 **慢–快双系统** 解耦长视界想象与高频修正；Streaming-WAM 按页面口径走的是 **动作前缀条件化** 补偿延迟，而非拆两套频率不同的模型 |
| [Fast-WAM](./paper-fast-wam.md) | **推理期跳过未来视频去噪** 直接压单次延迟；Streaming-WAM 不以砍预测为主，而是 **容忍延迟并与执行并行** |
| [DeltaWAM](./paper-deltawam.md) | 同期「改 WAM 节拍」条目：DeltaWAM 预测 **视觉 delta** 降单次推理成本；Streaming-WAM 改 **推理与控制的时序关系**，二者可正交组合（待验证） |

## 结论

**总判：Streaming-WAM 适合作为「World–Action 模型在推理延迟下用 action-conditioned 预测对齐即将发生的场景，使控制与 W…」方向的入口页；机制细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md) 对照，避免与其它 arXiv 混淆。
2. 开源为 **已开源** 时从项目页/GitHub 再核实一次再写复现计划。
3. 涉及异步 WAM 或恢复评测时，同时记录 **正常起点 SR** 与 **偏差后恢复率**（若适用）。

## 关联页面

- [具身研究 12 篇（恢复/WAM）技术地图](../overview/embodied-research-12-papers-recover-wam-technology-map.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/streaming_wam_arxiv_2609_28927.md)
- [Streaming-WAM 仓库归档](../../sources/repos/streaming-wam.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)

## 推荐继续阅读

- [arXiv:2609.28927](https://arxiv.org/abs/2609.28927)
- [GitHub 仓库](https://github.com/SJTU-DENG-Lab/Streaming-WAM)

