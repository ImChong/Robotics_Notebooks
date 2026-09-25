---
type: entity
tags:
  - paper
  - world-model
  - wam
  - replanning
  - execution
status: complete
updated: 2026-09-25
arxiv: "2608.09492"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ./paper-fast-wam.md
  - ./paper-memorywam.md
  - ../overview/embodied-frontier-algorithms-technology-map.md
sources:
  - ../../sources/papers/tempowam_arxiv_2608_09492.md
  - ../../sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md
summary: "TempoWAM（arXiv:2608.09492）：RPM 监测任务进度 + AEP 按需重规划，替换 WAM 固定 action-chunk 执行步数；真机易任务 WAM 推理 −26.9%，难任务成功率 +13.3 pp。"
---

# TempoWAM（arXiv:2608.09492）

**TempoWAM**（*Rethink Before You Execute: Adaptive Execution for World Action Models*，[arXiv:2608.09492](https://arxiv.org/abs/2608.09492)）来自 [机器人研发工程师 · 前沿算法盘点](../../sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)。

## 一句话定义

**RPM 监测任务进度 + AEP 按需重规划，替换 WAM 固定 action-chunk 执行步数；真机易任务 WAM 推理 −26.9%，难任务成功率 +13.3 pp。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TempoWAM | Timing Execution by Monitoring Progress Online | 本文自适应 WAM 执行方案 |
| WAM | World Action Model | 联合预测动作与环境演化 |
| RPM | Recurrent Progress Monitor | 循环进度监测模块 |
| AEP | Adaptive Execution Protocol | 自适应执行/重规划协议 |

## 为什么重要

- 固定前缀重规划与 chunk 可靠性随任务阶段变化不匹配；TempoWAM 是 plug-and-play 执行层而非重训骨干。
- 开源结论：**待发布**（步骤 2.5，2026-09-23）。
- 与 [具身前沿算法技术地图](../overview/embodied-frontier-algorithms-technology-map.md) 同路线条目可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.09492](https://arxiv.org/abs/2608.09492) |
| **开源** | **待发布** |
| **要点** | Recurrent Progress Monitor 估计进度；Adaptive Execution Protocol 决定继续执行或丢弃剩余 chunk；per-task 校准因子在线适配。 |
| **文内指标** | LIBERO / RoboTwin / 真机；易任务维持成功率下减推理次数，难任务抬成功。 |


## 源码运行时序图

**不适用**（入库日 2026-09-23：TempoWAM 为执行层插件，以论文/仓库 README 训练–推理入口为准；非单一可运行管线时序图）。


## 实验与评测

- LIBERO / RoboTwin / 真机；易任务维持成功率下减推理次数，难任务抬成功。
- **读法：** 索引级摘要；逐项 baseline 以原文 PDF 为准。

## 与其他工作对比

| 维度 | TempoWAM（本页） | 固定 action-chunk 执行的 WAM | [Fast-WAM](./paper-fast-wam.md) |
|------|-------------------|------------------------------|----------------------------------|
| 动的旋钮 | **执行多久才重规划**（RPM 进度 + AEP 按需丢弃剩余 chunk） | 固定步数 | **推理期要不要想象未来** |
| 易任务 | 维持成功率同时 **推理次数 −26.9%**（真机口径） | 推理次数恒定 | 单次推理更快 |
| 难任务 | 成功率 **+13.3 pp** | 基线 | 未在此口径下对齐 |
| 部署形态 | **执行层插件**，可叠加在既有 WAM 上 | — | 改训练/推理结构 |

- **它省的是「无谓的重规划」和「该改不改」两端：** 固定 chunk 长度在易任务上过度重规划（浪费算力），在难任务上又坚持执行已经失效的计划（丢成功率）；进度监测把这两类错误一起消掉，所以两端指标能同时变好。
- **与 Fast-WAM / [MemoryWAM](./paper-memorywam.md) 可叠加而非互斥：** 三者分别动 **执行节奏 / 推理结构 / 历史记忆**，属 [World-Action Models](../concepts/world-action-models.md) 谱系的不同层；把它们的数放同一张表比大小没有意义。
- **复现边界：** 代码 **待发布**（入库日口径），per-task 校准因子如何在线适配是复现的关键细节，**以 [原文 PDF](https://arxiv.org/abs/2608.09492) 为准**。

## 结论

**TempoWAM 把 WAM 部署瓶颈从「预测多准」部分转成「何时重规划」；入库日未见独立官方仓库。**

1. 开源边界：**待发布** — 以项目页/仓库实际链接为准（入库日 2026-09-23）。
2. 核心机制：Recurrent Progress Monitor 估计进度；Adaptive Execution Protocol 决定继续执行或丢弃剩余 chunk；per-task 校准因子在线适配。…
3. 部署前核对硬件栈与评测协议，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [paper-fast-wam](./paper-fast-wam.md)
- [paper-memorywam](./paper-memorywam.md)

## 参考来源

- [tempowam_arxiv_2608_09492.md](../../sources/papers/tempowam_arxiv_2608_09492.md)
- [wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md](../../sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)
- [arXiv:2608.09492](https://arxiv.org/abs/2608.09492)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2608.09492)

