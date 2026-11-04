---
type: comparison
tags: [path-planning, sampling-based, rrt, motion-planning, comparison]
status: complete
updated: 2026-11-04
related:
  - ../methods/rrt.md
  - ../methods/rrt-connect.md
  - ../methods/rrt-star.md
  - ../methods/informed-rrt-star.md
  - ../concepts/configuration-space.md
  - ../entities/ompl.md
  - ../entities/moveit2.md
  - ./robot-path-planning-five-paradigms-taxonomy.md
sources:
  - ../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md
summary: "RRT / RRT-Connect / RRT* / Informed RRT* 选型对照：前两者求可行解（单树 vs 双树），后两者求渐近最优（全域 Rewiring vs 找到初解后只在可改进子集采样）；都需按任务补碰撞检测、运动约束与轨迹后处理。"
---

# RRT 家族选型对照：RRT / RRT-Connect / RRT* / Informed RRT*

> 依据 [深蓝AI《彻底搞懂RRT与RRT*》](https://mp.weixin.qq.com/s/EeLwGT_pBUoEnIlsQObYdg)（2026-11-04）整理，并对照一手资料：OMPL 官方类文档（RRT / RRTConnect / RRTstar / InformedRRTstar）、LaValle《Planning Algorithms》§5.5.3、Karaman & Frazzoli（[arXiv:1105.1186](https://arxiv.org/abs/1105.1186)）与 Gammell 等（[arXiv:1404.2334](https://arxiv.org/abs/1404.2334)）的摘要（2026-11-04 核对）。Kuffner & LaValle 2000 原文 PDF 文本不可机读，RRT-Connect 细节以 OMPL 文档与 LaValle 教材为准。

## 一句话定义

**同一棵「随机采样 + 最近节点扩展 + 碰撞检测」的树，四个变体的分歧在于有限算力花在哪里：RRT / RRT-Connect 求尽快找到可行路，RRT* / Informed RRT* 在此之上持续把路径代价往最优逼近。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RRT | Rapidly-exploring Random Tree | 快速探索随机树，单树、求可行解 |
| RRT-Connect | RRT-Connect（Kuffner & LaValle, ICRA 2000） | 起点、终点各长一棵树并尝试连接 |
| RRT* | Optimal RRT | 选父节点 + Rewiring，渐近最优 |
| Informed RRT* | Informed RRT* | 找到初解后只在可改进子集内采样 |
| OMPL | Open Motion Planning Library | 开源采样规划库，四者均有实现 |
| C-space | Configuration Space | 配置空间，规划所在的状态空间 |

## 为什么重要

- 四个名字常被并列成「RRT 系列」，但它们回答的问题不同：前两者回答「有没有路」，后两者回答「路能不能更短」。选错会出现「要实时可行解却开了 RRT*」或「要最短路却只跑 RRT」的错配。
- 在 [OMPL](../entities/ompl.md) / [MoveIt 2](../entities/moveit2.md) 里它们只是规划器名字的切换，接口相同，因此选型主要靠理解差异而不是工程成本。

## 核心对比

| 维度 | [RRT](../methods/rrt.md) | [RRT-Connect](../methods/rrt-connect.md) | [RRT*](../methods/rrt-star.md) | [Informed RRT*](../methods/informed-rrt-star.md) |
|------|-----|-------------|------|---------------|
| 树的数量 | 单树（从起点） | 双树（起点 + 终点） | 单树 | 单树 |
| 目标 | 可行解 | 可行解，且通常更快 | 渐近最优 | 渐近最优，收敛更快 |
| 找到解后 | 停止（不再改连接） | 停止 | 继续采样 + Rewiring 改进 | 只在可改进子集采样、剪枝 |
| 理论保证 | 概率完备 | 概率完备 | 概率完备 + 渐近最优 | 与 RRT* 相同 |
| 每步额外开销 | 最低 | 低（多一次向另一棵树的扩展） | 近邻查询、父节点比较与重复碰撞检测 | 同 RRT*，加子集采样 |
| 典型难点 | 狭窄通道、路径锯齿 | 两棵树需能相互连通 | 有限时间仍非最优 | 可直接采样的子集依赖代价函数（路径长度下为长椭球体） |

### 逐对辨析

- **RRT vs RRT-Connect：** OMPL 文档给出的核心思路是「从起点和终点各长一棵 RRT 并尝试连接」。LaValle 教材 §5.5.3 指出双向搜索通常性能好得多，尤其利于逃出 bug trap；实现上一棵树新增节点后，另一棵树以该新节点为目标扩展，并可通过「总扩展较小的树」保持平衡。两者都不优化代价。
- **RRT vs RRT\*：** Karaman & Frazzoli 证明常用采样算法的解代价在温和条件下几乎必然收敛到**非最优**值，而 RRT* 几乎必然收敛到最优，计算复杂度与 RRT 相差常数因子。OMPL 文档同样把最优性定义为相对用户设定的 OptimizationObjective，并允许在代价低于阈值时提前终止。
- **RRT\* vs Informed RRT\*：** Gammell 等指出 RRT* 渐近求出起点到规划域内**每个**状态的最优路径，与单查询任务不符；对路径长度代价，可改进当前解的状态子集是长椭球体，不直接采样它时，大空间或高维下改进解的概率会变得任意小。Informed RRT* 是对 RRT* 的简单修改，保留相同的完备性与最优性保证，论文实验报告收敛速度与终解代价优于 RRT*。
- **RRT-Connect vs RRT\*：** 文章的归纳是取舍问题——有限算力优先「发现新路径」还是「改进已有路径」。两者并不互斥，工程上可先用可行解规划器拿到初解，再交给优化或平滑环节。

## 如何选型？

| 场景 | 建议起点 | 理由 |
|------|----------|------|
| 机械臂抓放、只需无碰撞即可 | RRT-Connect | 求可行解且通常比单树快；之后再做平滑与时间参数化 |
| 教学 / 作为基线对照 | RRT（带目标偏置） | 结构最简单；LaValle 教材提醒偏置过强会变得像贪心势场法，过弱则缺少连向目标的动力 |
| 路径长度 / 能耗有明确要求，可接受持续计算 | RRT* | 解随采样数改进；需设时间预算并接受有限时间非最优 |
| 同上，且空间大或维数高 | Informed RRT* | 找到初解后把采样集中到可改进子集，减少无效探索 |

## 常见误区

1. **「RRT* 全面取代 RRT」**：RRT* 每步更贵，只需可行解时这部分算力是浪费；文章配图对比仅为特定随机种子和预算下的示例。
2. **「渐近最优 = 拿到最优解」**：这是采样数趋于无穷时的概率意义结论，有时间预算的实际运行仍可能非最优。
3. **「双树一定更快」**：LaValle 教材的表述是双向搜索「通常」好得多，且要专门处理才能让两棵树接上并保持快速探索；带运动约束时从终点一侧生长的扩展方式需另行核对，不宜直接套用几何版。
4. **「规划器输出可直接下发」**：四者输出的都是路径，仍需连续碰撞检测、平滑、时间参数化与反馈控制（见 [RRT](../methods/rrt.md)「部署要补上的东西」）。

## 关联页面

- [RRT](../methods/rrt.md)、[RRT-Connect](../methods/rrt-connect.md)、[RRT*](../methods/rrt-star.md)、[Informed RRT*](../methods/informed-rrt-star.md)
- [配置空间](../concepts/configuration-space.md)
- [OMPL](../entities/ompl.md)、[MoveIt 2](../entities/moveit2.md)
- [路径规划五范式分类](./robot-path-planning-five-paradigms-taxonomy.md) — 采样范式在整个规划谱系中的位置

## 参考来源

- [深蓝AI《彻底搞懂RRT与RRT*》文章归档](../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md)
- OMPL 官方类文档：[RRT](https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRT.html)、[RRTConnect](https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTConnect.html)、[RRTstar](https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTstar.html)、[InformedRRTstar](https://ompl.kavrakilab.org/classompl_1_1geometric_1_1InformedRRTstar.html)
- LaValle, *Planning Algorithms*, §5.5.3 [Single-tree search](https://lavalle.pl/planning/node236.html) / [Balanced, bidirectional search](https://lavalle.pl/planning/node237.html)
- Kuffner & LaValle, *RRT-Connect: An Efficient Approach to Single-Query Path Planning*, ICRA 2000, DOI: 10.1109/ROBOT.2000.844730
- Karaman & Frazzoli, *Sampling-based Algorithms for Optimal Motion Planning*, IJRR 2011, [arXiv:1105.1186](https://arxiv.org/abs/1105.1186)
- Gammell, Srinivasa, Barfoot, *Informed RRT\**, [arXiv:1404.2334](https://arxiv.org/abs/1404.2334)
