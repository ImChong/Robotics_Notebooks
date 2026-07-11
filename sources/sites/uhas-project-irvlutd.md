# UHAS — Cross-Embodiment Robot Manipulation（IRVL UTD）

> 来源归档（ingest）

- **标题：** Cross-Embodiment Robot Manipulation via a Unified Hand Action Space
- **类型：** site（官方项目页）
- **发布方：** Intelligent Robotics and Vision Lab（IRVL），University of Texas at Dallas；合作方 Intelligent Robotics and Interactive Systems Lab（ASU）
- **原始链接：** <https://irvlutd.github.io/UHAS/>
- **页面标注：** RSS 2026 Dexterous Manipulation Workshop
- **入库日期：** 2026-07-11
- **一句话说明：** 官方对外页：以 **规范球面形变** 统一 Allegro / LEAP / Shadow / MANO 四手动作空间，展示 **单策略同时控制四只不同手** 的演示视频，以及仿真/真机手内立方体重定向结果与代码入口。

## 摘录要点（与论文分工）

- **核心图示：** 规范球 → 形变 → CIK → 多 embodiment 关节配置；URDF 自动建球、统一手表面稠密对应、驱动平面/向量参数化形变场。
- **实验展示：** 仿真四手 Repose Cube；真机 LEAP / Allegro 手内重定向对比（Multi-Hand / Single-Hand / Zero-shot / Joint baseline 的 MEAN Reposes）。
- **演示视频：** 项目页嵌入视频，展示 **一个策略并行控制四只运动学与指数量不同的手**——与 ingest 用户提供的卖点一致。
- **代码：** 页面提供 UHAS 代码仓库链接（以项目页当前条目为准）。
- **资助：** NSF 2346528、2520553；NVIDIA Academic Grant；XPeng gift。

## 对 wiki 的映射

- [UHAS](../../wiki/methods/uhas-unified-hand-action-space.md) — 面向读者的演示链接、四手同策略可视化与 workshop 语境
