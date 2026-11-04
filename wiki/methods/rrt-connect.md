---
type: method
tags: [path-planning, sampling-based, rrt, bidirectional]
status: complete
updated: 2026-11-04
related:
  - ./rrt.md
  - ./rrt-star.md
  - ./informed-rrt-star.md
  - ../concepts/configuration-space.md
  - ../entities/ompl.md
  - ../entities/moveit2.md
  - ../methods/a-star.md
  - ../comparisons/robot-path-planning-five-paradigms-taxonomy.md
sources:
  - ../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md
summary: "RRT-Connect 从起点和终点两侧同时生长树并尝试把它们连接起来，强调更快找到可行解；是 OMPL 等库中常见的 RRT 变体（本页仅整理文章与 OMPL README 的表述）。"
---

# RRT-Connect

> 依据 [深蓝AI《彻底搞懂RRT与RRT*》](https://mp.weixin.qq.com/s/EeLwGT_pBUoEnIlsQObYdg)；文章仅一句话介绍。OMPL 仓库 README 把 RRT-Connect 列在其 40 多种采样规划算法中（已读 README）。**原始论文本次未核对。**

## 一句话定义

**RRT-Connect 从起点和终点两侧各生长一棵树，并努力把两棵树连接起来，常用于强调可行解搜索的运动规划。**（文章表述）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RRT | Rapidly-exploring Random Tree | 快速探索随机树 |
| Bidirectional | Bidirectional search | 双向搜索（本页对「两侧同时生长」的称呼） |

## 为什么重要

- 在文章的框架里，它与 RRT* 代表两种取舍：RRT-Connect 偏「更快发现可行解」，RRT* 偏「改进已有路径」。

## 主要技术路线

- 两棵树分别以起点和终点为根；尝试把两棵树连接起来（文章表述）。
- 与 [RRT](./rrt.md) 一样需要状态空间、碰撞检测与运动约束处理（见该页「部署要补上的东西」）。
- 具体连接策略、复杂度与完备性性质**未在本页核对**，请以原始论文与 [OMPL](../entities/ompl.md) 文档为准。

## 关联页面

- [RRT](./rrt.md)、[RRT*](./rrt-star.md)、[Informed RRT*](./informed-rrt-star.md)
- [配置空间](../concepts/configuration-space.md)
- [OMPL](../entities/ompl.md)、[MoveIt 2](../entities/moveit2.md)
- [A*](./a-star.md)
- [路径规划五范式分类](../comparisons/robot-path-planning-five-paradigms-taxonomy.md)

## 参考来源

- [文章归档](../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md)
- [OMPL 库节点（README 列出 RRT-Connect）](../entities/ompl.md)
