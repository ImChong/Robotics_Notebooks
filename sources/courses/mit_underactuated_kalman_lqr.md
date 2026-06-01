# MIT — Underactuated Robotics & Optimal Control（KF / LQR / DDP）

> 来源归档（ingest）

- **标题：** MIT 欠驱动机器人与最优控制课程（估计 + LQR + DDP 模块）
- **类型：** course
- **主讲：** Russ Tedrake
- **入库日期：** 2026-06-01
- **链接：**
  - [Underactuated Robotics 课程站](https://underactuated.csail.mit.edu/)
  - [Ch.16 Estimation](https://underactuated.csail.mit.edu/estimation.html)
  - [Optimal Control 2025 播放列表](https://www.youtube.com/playlist?list=PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI)（Lec.8 LQR、Lec.12 DDP、Lec.21 Kalman & duality）

## 为什么值得保留

- 将 **KF / EKF、LQR、DDP/iLQR、MPC** 放在同一教学体系，是机器人方向「控制–估计」联合学习的一手课程入口。

## 核心模块摘录

| 模块 | 内容要点 | Wiki 映射 |
|------|----------|-----------|
| Estimation | Bayes 滤波 → KF；EKF 应用与局限 | [state-estimation](../../wiki/concepts/state-estimation.md)、[ekf](../../wiki/formalizations/ekf.md) |
| LQR (Lec.8) | DP / Pontryagin / Riccati 三种视角 | [lqr](../../wiki/formalizations/lqr.md) |
| DDP (Lec.12) | 二阶展开、与 iLQR 关系 | [lqr-ilqr](../../wiki/methods/lqr-ilqr.md) |
| Kalman duality (Lec.21) | 估计–控制对偶、LQG | [kalman-filter](../../wiki/formalizations/kalman-filter.md)、[lqr](../../wiki/formalizations/lqr.md) |

## 对 wiki 的映射

- [kalman-filter](../../wiki/formalizations/kalman-filter.md)
- [lqr](../../wiki/formalizations/lqr.md)
- [lqr-ilqr](../../wiki/methods/lqr-ilqr.md)
- [optimal-control](../../wiki/concepts/optimal-control.md)

## 当前提炼状态

- [x] 课程模块与 wiki 映射表
- [ ] 后续可补：各讲义的 PDF 版本号与年度差异注记
