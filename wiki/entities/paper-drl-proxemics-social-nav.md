---
type: entity
tags: [paper, social-navigation, deep-rl, proxemics, reward-shaping]
status: complete
updated: 2026-08-19
arxiv: "2608.12917"
related:
  - ./paper-nav-ps-balance.md
  - ./paper-pgif-mppi.md
  - ./paper-icrowdnav.md
  - ../methods/ppo.md
  - ../overview/navigation-slam-autonomy-stack.md
sources:
  - ../../sources/papers/drl_proxemics_arxiv_2608_12917.md
  - ../../sources/sites/drl-proxemics.md
  - ../../sources/blogs/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md
summary: "DRL Proxemics（arXiv:2608.12917）：Hall 近体学径向高斯混合场作 DRL 社会导航密集奖励；多密度下社会指标升且效率仍 competitive。截至入库日项目页无代码。"
---

# DRL Proxemics：把「别贴太近」写进可学习奖励

**DRL Proxemics**（*Towards Socially Compliant Navigation in Deep Reinforcement Learning via Proxemics-Based Reward Modeling*；[arXiv:2608.12917](https://arxiv.org/abs/2608.12917)，[项目页](https://drl-proxemics.github.io/)）用 **Hall proxemics** 把每个人的个人空间建模为径向高斯混合场，在机器人视野内算 **局部社会代价**，接入已有 DRL 导航方法。

## 一句话定义

**社会导航的关键，是把人的舒适边界变成密集、可微、可复用的 reward 信号，而不是事后调 collision penalty。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DRL | Deep Reinforcement Learning | 深度强化学习导航策略 |
| Proxemics | Proxemics (Hall) | 人际距离/个人空间学 |
| GMM | Gaussian Mixture Model | 径向高斯混合建模个人空间 |
| SR | Success Rate | 到达目标成功率 |
| CR | Collision Rate | 与人/障碍碰撞率 |

## 为什么重要

- **到达 ≠ 社会可接受：** 贴人太近能到目标，但部署不可接受。
- **密集社会信号可插拔：** 奖励模块可接到多种已有 DRL 导航 backbone。
- **与约束分解路线对照：** 相对 [nav-ps-balance](./paper-nav-ps-balance.md) 的 cost 阈值，本文走 **proxemics 形状场**。

## 核心信息

| 项 | 内容 |
|----|------|
| **出处** | arXiv:2608.12917（2026-08） |
| **方法** | 视野内 per-person 高斯混合社会代价 + 标准导航目标 |
| **评测** | 多人群密度、奖励基线、DRL 方法横评 |
| **开源（截至 2026-08-19）** | **未开源**：项目页 Code 链为占位 |

## 核心原理

```mermaid
flowchart LR
  ped["行人位置/速度"]
  field["Proxemics 高斯混合场"]
  cost["局部社会代价"]
  drl["DRL 导航策略"]
  ped --> field --> cost --> drl
```

## 工程实践

| 项 | 建议 |
|----|------|
| 源码运行时序图 | **不适用**（无公开代码） |
| 读表 | 必须同时看社会指标与路径效率 |
| 对照 | 与 [PGIF-MPPI](./paper-pgif-mppi.md) 的预测场、[nav-ps-balance](./paper-nav-ps-balance.md) 的 cost 分解对照读 |

## 结论

**Proxemics 给 DRL 社会导航提供了可解释的密集奖励模板。**

1. **形状场比单一距离阈值更细** — 能表达方向性舒适区。
2. **插件式设计** — 重点在 reward，不在换整套 planner。
3. **代码未开** — 数字以论文/项目页为准。
4. **部署仍要真机** — 仿真社会指标升不等于 crowd 接受。

## 局限与风险

- 无公开实现，超参与场参数不可复现。
- 高斯混合是否覆盖文化/场景差异未在本文档展开。
- 与 ORCA 等经典方法的对照需读原文表格。

## 与其他工作对比

相对 [nav-ps-balance](./paper-nav-ps-balance.md) 的 **cost 分解**：本文用 **proxemics 形状场** 作密集 reward。相对 [PGIF-MPPI](./paper-pgif-mppi.md)：后者偏 **预测场 + 采样规划**，本文偏 **RL reward 插件**。

## 实验与评测

摘要报告在多种人群密度、奖励基线与 DRL 方法下 **社会指标稳定提升**，同时导航效率仍 competitive。具体表格以论文为准（代码未开源）。

## 与其他工作对比

见上节 [nav-ps-balance](./paper-nav-ps-balance.md) / [PGIF-MPPI](./paper-pgif-mppi.md) 对照。

## 关联页面

- [世界模型与真实执行 10 篇技术地图](../overview/world-model-exec-10-papers-technology-map.md)
- [接近–安全跟随](./paper-nav-ps-balance.md)
- [PGIF-MPPI](./paper-pgif-mppi.md)
- [iCrowdNav](./paper-icrowdnav.md)
- [PPO](../methods/ppo.md)

## 参考来源

- [DRL Proxemics 论文摘录](../../sources/papers/drl_proxemics_arxiv_2608_12917.md)
- [项目页归档](../../sources/sites/drl-proxemics.md)
- [具身智能小站 10 篇盘点（2026-08-19）](../../sources/blogs/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md)

## 推荐继续阅读

- [arXiv:2608.12917](https://arxiv.org/abs/2608.12917)
- [DRL Proxemics 项目页](https://drl-proxemics.github.io/)
