---
type: method
tags: [path-planning, sampling-based, rrt, asymptotic-optimality]
status: complete
updated: 2026-11-04
related:
  - ./rrt.md
  - ./rrt-connect.md
  - ./informed-rrt-star.md
  - ../concepts/configuration-space.md
  - ../entities/ompl.md
  - ../entities/moveit2.md
  - ../methods/a-star.md
  - ../comparisons/robot-path-planning-five-paradigms-taxonomy.md
  - ../comparisons/rrt-family-variants.md
sources:
  - ../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md
summary: "RRT* 在 RRT 基础上为新节点选累计代价最低的父节点并做 Rewiring，使路径代价随采样渐近收敛到最优；代价是更多近邻查询与碰撞检测，有限时间内仍可能非最优。"
---

# RRT*

> 依据 [深蓝AI《彻底搞懂RRT与RRT*》](https://mp.weixin.qq.com/s/EeLwGT_pBUoEnIlsQObYdg) 与一手论文 Karaman & Frazzoli《Sampling-based Algorithms for Optimal Motion Planning》（[arXiv:1105.1186](https://arxiv.org/abs/1105.1186)，IJRR 2011，摘要页已核对）整理。

## 一句话定义

**RRT* 保留 RRT 的随机扩展思想，同时把路径代价纳入搜索：新节点选累计代价更低的父节点，并通过 Rewiring 改写旧节点的连接，让已找到的连接持续变好。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RRT* | Optimal RRT | 渐近最优的 RRT 变体 |
| PRM* | Optimal Probabilistic Roadmap | 同篇论文提出的渐近最优 PRM |
| Rewiring | Rewiring | 重新连边：改写旧节点的父节点 |

## 为什么重要

- 一手摘要：采样规划算法（PRM、RRT）有概率完备性，但此前缺少对解质量的形式化分析；论文证明广泛使用的采样算法的解代价在温和条件下几乎必然收敛到**非最优值**，并提出 PRM* 与 RRT*——解代价几乎必然收敛到最优，且计算复杂度与其概率完备的对应算法相差常数因子。

## 主要技术路线

### RRT* 比 RRT 多做的事（文章）

1. **选父节点**：检查新节点附近的多个候选父节点，选「从起点经它到达新节点累计代价最低且连接无碰撞」的那个，未必是几何最近。
2. **Rewiring**：新节点可能给某个旧节点提供更便宜的连接；若总成本下降且新连接无碰撞，就改旧节点的父节点，受影响后代的累计代价同步更新。是局部调整，不是推倒重来。

### 必须说准的一点

- **渐近最优 ≠ 有限时间内全局最优**：理论结论是满足空间、采样、连接半径等条件、采样数趋于无穷时，最优解代价以概率意义收敛到最优；实际有时间预算，拿到的可能仍非最优。

## RRT 还是 RRT*？（文章）

- 只要尽快找到可行路径：基础 RRT 的思路直接；找到第一条后愿意继续投入计算：RRT*。
- RRT* 要查询更多近邻、比较父节点累计代价并重复碰撞检测与 Rewiring，复杂空间里最贵的往往是验证连接是否可行，而非采样。
- RRT* **不是**全面取代 RRT；文章配图的对比只是特定随机种子与计算预算下的示例，不能概括所有任务的速度与性能。
- 变体围绕同一问题：有限算力优先「发现新路径」还是「改进已有路径」（见 [RRT-Connect](./rrt-connect.md)、[Informed RRT*](./informed-rrt-star.md)）。

## 关联页面

- [RRT](./rrt.md)
- [RRT-Connect](./rrt-connect.md)、[Informed RRT*](./informed-rrt-star.md)
- [A*](./a-star.md)
- [配置空间](../concepts/configuration-space.md)
- [OMPL](../entities/ompl.md)、[MoveIt 2](../entities/moveit2.md)
- [路径规划五范式分类](../comparisons/robot-path-planning-five-paradigms-taxonomy.md)
- [RRT 家族选型对照](../comparisons/rrt-family-variants.md) — RRT / RRT-Connect / RRT* / Informed RRT* 并排对比

## 参考来源

- [文章归档](../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md)
- Karaman & Frazzoli, [arXiv:1105.1186](https://arxiv.org/abs/1105.1186)（提交 2011-05-05）
- OMPL RRTstar 类文档：<https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTstar.html>
