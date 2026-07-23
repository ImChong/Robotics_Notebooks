# FAR Planner（far_planner）

> 来源归档

- **标题：** FAR Planner: Fast, Attemptable Route Planner using Dynamic Visibility Update
- **类型：** repo
- **链接：** https://github.com/MichaelFYang/far_planner
- **论文：** https://arxiv.org/abs/2110.09460（IROS 2022）
- **项目页：** https://www.cmu-exploration.com/
- **机构：** Carnegie Mellon University Robotics Institute（CMU）
- **入库日期：** 2026-07-23
- **开源状态：** **已开源**
- **一句话说明：** 基于动态可见图（visibility graph）的快速可尝试路径规划：多边形障碍表示、局部/全局双层图更新，支持已知图与未知环境 attemptable 导航。
- **沉淀到 wiki：** [far-planner](../../wiki/entities/far-planner.md)、[autonomous-exploration](../../wiki/tasks/autonomous-exploration.md)

---

## 核心定位

**FAR**（Fast, Attemptable Route）在导航过程中维护 **可见图**：从距离传感器提取障碍边缘点 → 封闭多边形 → 可见边；一线程增量更新图（约单核 20%），另一线程毫秒级搜路。未知环境可在自由空间不可达时 **尝试穿越未知区**；动态障碍则断开/恢复被遮挡的可见边。

与 [TARE Planner](tare_planner.md) 互补：TARE 产出探索路点/路径，FAR 负责 **长距离快速重规划**。二者集成于 [CMU Autonomous Exploration Development Environment](../sites/cmu-exploration.md)。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 方法实体 | [far-planner](../../wiki/entities/far-planner.md) |
| A\* 对照 | [a-star](../../wiki/methods/a-star.md) |
| 自主探索 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 课程映射 | [humanoid-system-curriculum](../../wiki/entities/humanoid-system-curriculum.md) Ch5 |
