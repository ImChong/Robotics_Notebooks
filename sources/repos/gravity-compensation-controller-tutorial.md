# PAL gravity_compensation_controller_tutorial

> 来源归档

- **标题：** gravity_compensation_controller_tutorial
- **类型：** repo
- **来源：** PAL Robotics（Adria Roig）
- **链接：** <https://github.com/pal-robotics/gravity_compensation_controller_tutorial>
- **文档站：** <https://docs.pal-robotics.com/25.01/hardware/controllers/gravity-compensation.html>
- **许可：** **未声明**（GitHub API `license=null`；`package.xml` 为 `<license>TODO</license>`；仓内无 `LICENSE`）
- **Stars：** 20（2026-08-13）
- **语言：** C++
- **入库日期：** 2026-08-13
- **一句话说明：** ros_control 力矩接口上的重力补偿教程：RBDL `InverseDynamics(q,0,0)` 算 $g(q)$，再叠加静/粘摩擦并除以 $K_t\cdot N$ 下发电流/力矩。
- **沉淀到 wiki：** [`wiki/concepts/gravity-compensation.md`](../../wiki/concepts/gravity-compensation.md)

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| GitHub | 非 fork；默认分支 `master`；最后推送 2021-12；约 20 stars |
| 可运行入口 | `src/gravity_compensation_controller.cpp`；`roslaunch ... robot:=tiago`；`test/rrbot_gravity_controller_test.cpp` |
| 动力学核 | RBDL `RigidBodyDynamics::InverseDynamics(model, q, 0, 0, tau)` ≡ $g(q)$ |
| 真机 | 面向 [TIAGo](http://wiki.ros.org/Robots/TIAGo) 7 轴臂 + pal-gripper / hey5 / schunk-wsg |
| 生产对照 | PAL OS 25.01 文档中的 `pal_controllers/GravityCompensationController` **不在本仓**，属闭源发行版 |

**结论：部分开源。** 教程与 RRBot 测试公开可编译；生产控制器与许可均未开放/未声明。复用前需自行确认授权，不要把它当成 PAL 量产控制器的源码。

## 控制律（`update()`）

```
τ_g = InverseDynamics(q, 0, 0)
τ = τ_g + b q̇ + τ_c · sign_tol(q̇)
effort = τ / (K_t · N)
```

- `viscous_friction` / `static_friction` / `velocity_tolerance` 可动态重配置。
- 仿真必须把 `motor_torque_constant` 与 `reduction_ratio` 设为 1.0，否则 Gazebo 与真机电流接口对不齐。
- README 强调：**停掉本控制器前必须切回位置臂控制器，否则手臂会落下。**

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [重力补偿](../../wiki/concepts/gravity-compensation.md) | 算法与工程清单 |
| [摩擦补偿](../../wiki/concepts/friction-compensation.md) | 本教程把摩擦叠在 $g(q)$ 上 |
| [Pinocchio](../../wiki/entities/pinocchio.md) / [Dynibo](../../wiki/entities/dynibo.md) | 同等 $g(q)$ 计算，不必绑 RBDL |
| [PAL 文档站](../sites/pal-robotics-gravity-compensation.md) | 生产 YAML（7 轴 TIAGo PRO） |
