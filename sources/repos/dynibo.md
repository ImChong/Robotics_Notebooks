# dynibo

> 来源归档

- **标题：** dynibo
- **类型：** repo
- **作者：** Xue Xiaojie（[xiaojie-xue](https://github.com/xiaojie-xue)）
- **链接：** https://github.com/xiaojie-xue/dynibo
- **PyPI：** https://pypi.org/project/dynibo/（`pip install dynibo`，包名 `dynibo`，v0.1.0）
- **Cargo：** `cargo add dynibo`
- **Stars：** ~18（2026-08-09）
- **入库日期：** 2026-08-09
- **许可证：** MIT
- **代码：** **已开源** — GitHub 仓库 + PyPI 轮子 + CMake C/C++ 绑定；Rust 核心 + Python / C / C++ 同一实现
- **一句话说明：** 快速、轻量的 Rust 树状机器人运动学与动力学库：运行时加载 URDF，`Workspace` 计算期零分配；公开 API 覆盖 FK / Jacobian / 速度加速度运动学 / DLS-IK / 重力补偿 / RNEA，并以 Pinocchio 作 oracle 与 benchmark。
- **沉淀到 wiki：** 是 → [`wiki/entities/dynibo.md`](../../wiki/entities/dynibo.md)

---

## 核心定位

- **问题：** 控制环与嵌入式侧常只需一小撮运动学/动力学原语（FK、Jacobian、重力、RNEA、数值 IK），但主流库（如 Pinocchio）生态重、依赖面宽；Rust 侧缺少「runtime URDF + 零分配 Workspace」的轻量选择。
- **做法：** Rust 核心（`nalgebra` + `urdf-rs`）解析树状 URDF → `Robot` + 可复用 `Workspace`；计算循环内不分配；Python wheel 捆绑原生库；C/C++ 走 CMake `dynibo::dynibo`。
- **公开接口（README）：** `forward_kinematics`、`jacobian`、`forward_velocity_kinematics`、`forward_acceleration_kinematics`、`inverse_kinematics`（阻尼最小二乘）、`gravity`（可外载）、`inverse_dynamics`（RNEA）。
- **性能声明（作者 Criterion，Pinocchio 3.9.0 对照）：** FK / Jacobian / Gravity / RNEA 约 **1.17–2.70×**；双叶树模型 Jacobian 最高约 **2.70×**（i9-14900K；计时不含模型/Workspace/输出分配）。
- **可靠性格局：** 有限差分运动学、动力学回归、树状/外载、IK、非法输入、Workspace 归属、计算期零分配；独立 Pinocchio oracle 对比 FK/Jacobian/gravity/RNEA；Rust 核心无项目自有 `unsafe`；CI 行覆盖 ≥85%、分支 ≥75%。
- **模型范围：** 运行时尺寸的树状 URDF；关节类型 revolute / continuous / prismatic / fixed；无效拓扑与长度不匹配返回结构化错误。
- **运行时序：** 见 [wiki/entities/dynibo.md § 源码运行时序图](../../wiki/entities/dynibo.md#源码运行时序图)。

---

## 仓内入口（2026-08 快照）

| 路径 | 角色 |
|------|------|
| `src/lib.rs` / `src/robot.rs` / `src/robot/workspace.rs` | Rust 核心：`Robot`、`Workspace`、运动学/动力学 |
| `src/urdf.rs` / `src/spatial.rs` | URDF 树解析与 `Frame` / `Twist` / `Wrench` |
| `bindings/python/` | PyO3 绑定；`pip install dynibo` |
| `bindings/c/` | C ABI + C++ 头 + CMake 包 |
| `examples/franka.rs` + `examples/data/franka_fer.urdf` | Franka 法兰 FK / Jacobian / gravity 示例 |
| `benches/pinocchio.rs` | 与 Pinocchio 对照的 Criterion bench（`--features pinocchio-bench`） |
| `tests/pinocchio_oracle.rs` | Pinocchio oracle 数值对照 |
| `ci/test-all.sh` | Rust + Pinocchio + Python + C/C++ 全套验证 |

---

## 对 wiki 的映射

- 实体页：[Dynibo](../../wiki/entities/dynibo.md)
- 动力学内核对照：[Articulated Body Algorithms](../../wiki/formalizations/articulated-body-algorithms.md)、[Pinocchio](../../wiki/entities/pinocchio.md)
- 模型入口：[URDF](../../wiki/concepts/urdf-robot-description.md)
- IK 分工：[ssik](../../wiki/entities/ssik.md)（解析全分支）vs Dynibo DLS 数值 IK
- 上手对照：[Pinocchio 快速上手](../../wiki/queries/pinocchio-quick-start.md)
