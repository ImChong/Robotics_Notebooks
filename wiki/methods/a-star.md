---
type: method
tags: [path-planning, search, navigation, graph-search, mobile-robot]
status: complete
updated: 2026-07-23
related:
  - ./dwa.md
  - ./smooth-navigation-path-generation.md
  - ../entities/python-robotics.md
  - ../entities/navigation2.md
  - ../concepts/dynamic-obstacle-filtering.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/repos/python_robotics.md
summary: "A* 全局路径规划：在栅格或图上用 g+h 启发搜索最短/最优路径，是移动机器人与课程第 4.2 节全局规划的经典基线。"
---

# A\* 全局路径规划

## 一句话定义

**A\*** 在离散图（常为占据栅格邻接）上搜索从起点到终点的路径，用 \(f = g + h\)（已走代价 + 启发估计）保证在可采纳启发下找到最优路径——课程第 4.2 节核心算法。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| A\* | A-star | 带启发的最优图搜索 |
| Dijkstra | Dijkstra's Algorithm | \(h=0\) 时的特例 |
| Heuristic | Heuristic Function | 常用欧氏/曼哈顿距离 |
| Grid Map | Occupancy Grid Map | 二维导航常用离散化 |
| Navfn / Nav2 | ROS Navigation planners | 工程中封装 A\*/变体 |

## 为什么重要

- 全局层给出 **拓扑可行折线**；局部 [DWA](./dwa.md) 再处理动力学与瞬态障碍。
- PythonRobotics / Nav2 均把它作为教学与工程默认全局器之一。
- 输入地图质量依赖 [动态障碍物滤波](../concepts/dynamic-obstacle-filtering.md)，否则静态层被瞬态占用污染。

## 主要技术路线

| 路线 | 说明 | 典型场景 |
|------|------|----------|
| 栅格 A\* | 占据栅格四/八连通 + 欧氏/曼哈顿 \(h\) | 课程 2D 导航、NavFn |
| 带转向代价 A\* | \(g\) 含航向变化惩罚 | 差速/Ackermann 全局层 |
| 混合 A\* / Smac | 运动学可行原语扩展 | Nav2 Smac Planner |
| 任意时刻 / D\* 族 | 动态障碍下增量重规划 | 环境频繁变化 |

## 核心原理

1. 维护 open/closed 集；每次扩展 \(f\) 最小节点。
2. \(g\)：从起点累计代价（可含距离、转向惩罚）。
3. \(h\)：到终点的低估距离（欧氏等）；不可高估，否则失去最优性。
4. 邻接：四/八连通；障碍格与膨胀层不可通行。

## 工程实践

- 教学： [PythonRobotics](../entities/python-robotics.md) Path Planning → A\* 动画。
- 工程： [Navigation2](../entities/navigation2.md) `planner_server`（NavFn/Smac 等）。
- 折线过糙时接 [路径平滑](./smooth-navigation-path-generation.md)。

## 局限与风险

- 高分辨率大地图内存与时间开销大；动态环境需重规划或 D\* 族。
- 忽略机器人动力学——必须与局部规划器配合。

## 关联页面

- [DWA](./dwa.md)
- [FAR Planner](../entities/far-planner.md) — 长距离可见图对照
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [PythonRobotics 归档](../../sources/repos/python_robotics.md)

## 推荐继续阅读

- Hart, Nilsson, Raphael, *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*, 1968
