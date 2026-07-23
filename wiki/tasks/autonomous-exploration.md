---
type: task
tags: [exploration, navigation, autonomy, lidar, planning, cmu]
status: complete
updated: 2026-07-23
related:
  - ../entities/tare-planner.md
  - ../entities/far-planner.md
  - ../methods/a-star.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../entities/paper-autonomous-spot-nebula-exploration.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/sites/cmu-exploration.md
summary: "机器人自主探索任务：在未知环境中主动选择视点/路径以最大化信息增益并保持可通行，课程以 TARE（探索）+ FAR（路由）为教学栈。"
---

# 机器人自主探索

## 一句话定义

**自主探索**要求机器人在 **无完整先验地图** 时主动规划观测与运动，尽快覆盖未知空间并维持定位与安全通行——课程第 5.1 / 5.4 节任务定义。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TARE | Technologies for Autonomous Robot Exploration | CMU 分层探索规划器 |
| FAR | Fast Attemptable Route Planner | CMU 可见图路由规划器 |
| TSP | Traveling Salesman Problem | 探索路点排序常用近似 |
| Frontier | Frontier | 已知自由与未知边界 |
| NBVP | Next-Best-View Planner | 经典下一最佳视点基线 |

## 为什么重要

- 区别于「给定目标点导航」：目标由 **信息增益 / 覆盖** 在线生成。
- 地下、废墟、大型室内巡检等场景的核心能力；课程用仿真把 TARE+FAR 跑通即可迁移思路到人形巡航。

## 核心原理

典型流水线：

1. 局部感知更新占据/点云地图。
2. 提取 frontier 或体素增益。
3. **探索规划器**（如 [TARE](../entities/tare-planner.md)）生成覆盖路径。
4. **路由/局部规划**（如 [FAR](../entities/far-planner.md) + 局部跟踪）执行并避障。
5. 循环直至覆盖完成或超时。

## 工程实践

- 入门环境：[CMU Exploration](../../sources/sites/cmu-exploration.md) + 官方仿真。
- 评价指标：覆盖体积/面积、路径长度、重访率、计算负载。
- 与 [Spot NeBula 探索论文](../entities/paper-autonomous-spot-nebula-exploration.md) 对照系统级差异。

## 局限与风险

- 多数开源栈面向地面车；人形需额外考虑 **可通行性与行走策略接口**。
- 探索完成判据与传感器 FOV 强相关，仿真成功 ≠ 真机覆盖。

## 关联页面

- [TARE Planner](../entities/tare-planner.md)
- [FAR Planner](../entities/far-planner.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [CMU Exploration 站点归档](../../sources/sites/cmu-exploration.md)

## 推荐继续阅读

- <https://www.cmu-exploration.com/>
