---
type: entity
tags: [paper, task-planning, neuro-symbolic, mobile-manipulation, iros-2026]
status: complete
updated: 2026-10-02
arxiv: "2606.06877"
related:
  - ../methods/trajectory-optimization.md
  - ../tasks/manipulation.md
  - ../overview/iros-2026-awards-9-papers-technology-map.md
sources:
  - ../../sources/papers/iflax_arxiv_2606_06877.md
  - ../../sources/blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md
summary: "iFlax（arXiv:2606.06877，IROS 2026 ReS AI WS 最佳论文）：在线双层优化学物体重要性 + 3R 符号规划恢复；MazeNamo 失败率 −80%、规划时间 −57%；Spot 移动操作验证；代码待发布。"
---

# iFlax（Imperative neuro-symbolic Flax）

**Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints**（[arXiv:2606.06877](https://arxiv.org/abs/2606.06877)，[项目页](https://sairlab.org/iflax/)，**IROS 2026 ReS AI Workshop 最佳论文**）提出 **iFlax**：把 **物体重要性学习** 写成 **imperative 双层优化**——上层神经网络在任务关系图上预测重要性，下层 **PDDL 符号规划器** 在剪枝搜索空间求解；并用 **Repair / Restart / Rollback（3R）** 给上层提供稳定反馈，缓解 **Flax 式 exposure bias**。

## 一句话定义

**长时程逻辑约束规划不能只靠离线全空间标签训练剪枝器——要让规划器自己的失败与恢复轨迹，在线教神经网络「哪些物体真的不能删」。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PDDL | Planning Domain Definition Language | 符号规划问题描述 |
| 3R | Repair, Restart, Rollback | 下层规划恢复策略 |
| WS | Workshop | IROS 研讨会论文 |
| Spot | Boston Dynamics Spot | 四足移动操作验证平台 |

## 为什么重要

- 纳入 [IROS 2026 九篇获奖盘点](../../sources/blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)。
- 论文报告：MazeNamo 上相对 prior SOTA **失败率 −80.04%**、**规划时间 −57.14%**。
- **开源结论（2026-10-02）：待发布** — 无公开 `sair-lab/iflax` 代码仓。

## 核心机制

| 项 | 内容 |
|----|------|
| **上层** | 图神经网络/关系网络预测 object-importance |
| **下层** | 剪枝后符号规划 + **3R** 并行恢复 |
| **训练信号** | 规划成败与搜索空间 **与部署一致** |

## 源码运行时序图

**不适用**（截至入库日仅有论文/视频，**无官方可运行代码仓库**。）

## 实验与评测

- 基准：MazeNamo、SokoMindPlus、LogisticsPlus 等。
- 真机/仿真：**Spot 移动操作** 长时程 MazeNamo 执行。

## 结论

**iFlax 代表 neuro-symbolic 规划从「静态剪枝器」走向「规划反馈闭环训练」** — 3R 是稳定 imperative learning 的关键工程件。

1. **Exposure bias** 是剪枝式 neuro-symbolic 的真实杀手；iFlax 用 **在线双层** 对齐 train/deploy 搜索空间。
2. **80% 失败率下降** 是 **MazeNamo + 对照 Flax** 口径；换基准需重读表格。
3. 与 **VAP-TAMP**（同盘点）互补：iFlax 偏 **离散任务规划剪枝**，VAP-TAMP 偏 **执行期 VLM 验证与重规划**。
4. 代码未开源前，复现重点在 **PDDL 域 + 3R 规划器接口**。

## 关联页面

- [IROS 2026 九篇获奖地图](../overview/iros-2026-awards-9-papers-technology-map.md)
- [VAP-TAMP](./paper-vap-tamp.md)
- [轨迹优化 / TAMP 方法页](../methods/trajectory-optimization.md)

## 参考来源

- [IROS 2026 九篇获奖盘点（公众号）](../../sources/blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)
- [iFlax sources 归档](../../sources/papers/iflax_arxiv_2606_06877.md)

## 推荐继续阅读

- 项目页：<https://sairlab.org/iflax/>
