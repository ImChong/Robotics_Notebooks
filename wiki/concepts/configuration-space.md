---
type: concept
tags: [kinematics, motion-planning, dof, modern-robotics, goodman-wechat]
status: complete
updated: 2026-10-01
related:
  - ../overview/modern-robotics-wechat-principles-series.md
  - ../formalizations/forward-kinematics.md
  - ../formalizations/homogeneous-coordinates-transform.md
  - ../entities/modern-robotics-book.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/wechat_goodman_cspace_robot_motion_planning.md
  - ../../sources/raw/wechat_modern_robotics_album_4521219024549937157/01_mid2247483809/01_mid2247483809.md
  - ../../sources/papers/modern_robotics_textbook.md
summary: "位形空间 C-space 把机器人所有合法位形当作一个抽象空间中的点；运动规划在该空间中寻路，而非仅在任务空间或笛卡尔坐标里。"
---

# 位形空间（Configuration Space, C-space）

**一句话：** 用最少独立坐标描述整台机器人的状态，所有可能位形的集合就是 C-space；规划器在 C-space 里找路径，碰撞与约束也首先在这里定义。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| C-space | Configuration Space | 机器人位形集合；维数 = 机构 dof |
| dof | Degrees of Freedom | 描述位形所需独立实数坐标个数 |
| SE(2) | Special Euclidean Group in 2D | 平面刚体位姿群，小车 $(x,y,\theta)$ 典型 |
| SE(3) | Special Euclidean Group in 3D | 空间刚体位姿群 |
| FK | Forward Kinematics | 关节坐标 → 末端位姿；连接 C-space 与任务空间 |

## 为什么重要

- **规划在 C-space 做**：碰撞检测、距离场、RRT/PRM 的采样点都是位形 $q$，不是单独的末端 $(x,y,z)$。
- **冗余与多解**：同末端位姿可对应多个 $q$（肘上/肘下）；忽略 C-space 会漏解或误判可达性。
- **约束类型不同**：完整约束减少 C-space 维数；非完整约束（如平面小车）限制速度但不改变可达位形维数。

## 核心结构

### 位形 vs 任务空间

| 概念 | 含义 |
|------|------|
| **位形** | 确定刚体上每一点位置所需的最小状态（含全部关节角） |
| **C-space** | 所有位形的集合；维数 = dof |
| **任务空间** | 末端或关注刚体的位姿/位置空间 |
| **工作空间** | 末端在物理空间中能到达的区域 |

[正运动学](../formalizations/forward-kinematics.md) 是从 C-space 到任务空间的映射；逆问题在任务空间给定目标，在 C-space 搜索 $q$。

### 自由度计数

Grübler 公式（空间机构）：$F = 6(l-1) - \sum_i (6-f_i)$，$l$ 为构件数（含地面），$f_i$ 为第 $i$ 关节允许相对自由度数。关键是 **独立约束** 只计一次。

平面 2R 开链：$F=2$；若两关节均可转满圈，C-space 同胚于环面 $S^1\times S^1$，常展开为 $[0,2\pi)^2$ 画图。

### 约束分类

- **完整（holonomic）**：$g(q)=0$ 限制可达位形，降低 C-space 维数。
- **非完整（nonholonomic）**：Pfaffian 形式 $A(q)\dot q=0$ 限制瞬时速度；平面小车不能侧滑是典型例，可达 C-space 仍为 3 维 SE(2)。

## 流程总览

```mermaid
flowchart LR
  Q["关节/广义坐标 q"]
  C["C-space 路径规划"]
  FK["正运动学 T(q)"]
  W["任务/工作空间"]
  Q --> C --> FK --> W
```

## 常见误区

- 把 **电机数** 或 **关节数** 直接当作 dof，忽略闭链与冗余约束。
- 在 **任务空间** 里做 RRT 却用 FK 投影回关节，而不在 C-space 采样（易撞中间连杆）。
- 认为 **欧拉角三个数** 全局无奇异地参数化 SE(3)（局部可用，全局需群/四元数）。

## 关联页面

- [Modern Robotics 微信精读系列](../overview/modern-robotics-wechat-principles-series.md)
- [齐次坐标与 SE(3)](../formalizations/homogeneous-coordinates-transform.md)
- [Modern Robotics 教材](../entities/modern-robotics-book.md)

## 参考来源

- [从真实空间到 C-space（公众号精读）](../../sources/blogs/wechat_goodman_cspace_robot_motion_planning.md)
- [Modern Robotics 教材归档](../../sources/papers/modern_robotics_textbook.md)

## 推荐继续阅读

- [Northwestern Modern Robotics Ch 2 PDF](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf) — C-space 与约束正式定义
- [深蓝《具身智能基础》几何专栏](../overview/shenlan-embodied-ai-fundamentals-series.md) — 互补的 L0 几何链
