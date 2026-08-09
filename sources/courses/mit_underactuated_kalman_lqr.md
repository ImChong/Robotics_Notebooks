# MIT — Underactuated Robotics（KF / LQR / DDP 相关模块）

> 来源归档（ingest）

- **标题：** MIT Underactuated Robotics（估计 + LQR + DDP 相关章节）
- **类型：** course
- **主讲：** Russ Tedrake
- **入库日期：** 2026-06-01
- **更新日期：** 2026-08-09（澄清：勿与 CMU Optimal Control 2025 playlist 混淆）
- **链接：**
  - [Underactuated Robotics 课程站](https://underactuated.csail.mit.edu/)
  - [Ch.16 Estimation](https://underactuated.csail.mit.edu/estimation.html)

> **澄清（2026-08-09）：** YouTube 播放列表 [Optimal Control 2025](https://www.youtube.com/playlist?list=PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI) 属于 **CMU 16-745 / Zachary Manchester**，不是 Tedrake Underactuated。完整讲次归档见 [`cmu_optimal_control_16_745_2025_youtube.md`](./cmu_optimal_control_16_745_2025_youtube.md) 与 [`wiki/entities/cmu-optimal-control-curriculum.md`](../../wiki/entities/cmu-optimal-control-curriculum.md)。本文件只保留 Underactuated 课程自身入口。

## 为什么值得保留

- Tedrake Underactuated 将 **估计、LQR、轨迹优化 / 欠驱动** 放在同一教学体系，是机器人「控制–估计」联合学习的另一条一手课程入口；可与 CMU 16-745 交叉对照，而非互相替代。

## 核心模块摘录

| 模块 | 内容要点 | Wiki 映射 |
|------|----------|-----------|
| Estimation | Bayes 滤波 → KF；EKF 应用与局限 | [state-estimation](../../wiki/concepts/state-estimation.md)、[ekf](../../wiki/formalizations/ekf.md) |
| Underactuated / traj | 欠驱动动力学与轨迹优化视角 | [optimal-control](../../wiki/concepts/optimal-control.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |
| LQR / DDP（对照阅读） | 与 CMU 课 LQR/DDP 讲对照时，用本站 Underactuated 章节 + CMU 录像 | [lqr](../../wiki/formalizations/lqr.md)、[lqr-ilqr](../../wiki/methods/lqr-ilqr.md)；录像见 [CMU OC 2025](./cmu_optimal_control_16_745_2025_youtube.md) |

## 对 wiki 的映射

- [kalman-filter](../../wiki/formalizations/kalman-filter.md)
- [lqr](../../wiki/formalizations/lqr.md)
- [lqr-ilqr](../../wiki/methods/lqr-ilqr.md)
- [optimal-control](../../wiki/concepts/optimal-control.md)
- [cmu-optimal-control-curriculum](../../wiki/entities/cmu-optimal-control-curriculum.md) — CMU 16-745 完整 playlist 策展

## 当前提炼状态

- [x] 与 CMU Optimal Control 2025 playlist 解耦并互链
- [ ] 后续可补：Underactuated 各章 PDF/HTML 版本号与年度差异注记
