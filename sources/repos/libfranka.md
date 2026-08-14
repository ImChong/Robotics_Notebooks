# libfranka

> 来源归档

- **标题：** libfranka
- **类型：** repo
- **来源：** Franka Robotics（原 Franka Emika）
- **链接：** <https://github.com/frankarobotics/libfranka>
- **文档 / 项目页：** <https://frankarobotics.github.io/docs/doc/libfranka/docs/index.html>（GitHub `homepage`）
- **许可：** Apache-2.0
- **入库日期：** 2026-08-13
- **一句话说明：** Franka 七轴科研臂的实时 C++ 客户端；`examples/cartesian_impedance_control.cpp` 是官方笛卡尔阻抗 + 零空间关节阻抗示例，`generate_elbow_motion` 把第 7 自由度参数化为肘角。
- **沉淀到 wiki：** [`wiki/entities/franka-research-3.md`](../../wiki/entities/franka-research-3.md)、[`wiki/concepts/null-space-control.md`](../../wiki/concepts/null-space-control.md)

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| GitHub | `frankarobotics/libfranka`；C++；Apache-2.0；默认分支 `main` |
| 可运行入口 | `examples/` 可执行文件列表见 `examples/CMakeLists.txt` |
| 零空间相关 | `cartesian_impedance_control`（$\tau_{\mathrm{task}}+\tau_{\mathrm{nullspace}}$）；`generate_elbow_motion`（笛卡尔运动生成器的 elbow 参数 = 7 轴自运动） |
| 配套 | [franka_ros](https://github.com/frankarobotics/franka_ros) 的 `cartesian_impedance_example_controller` 是同一公式的 ROS 包装 |

**结论：已开源。** 控制器只覆盖 **Franka 机型**；要多机型零空间阻抗用 [Cartesian-Impedance-Controller](./cartesian-impedance-controller.md)。

## 7 轴读法

Franka Panda / FR3 为 **7R**，笛卡尔 6D 任务剩 **1 维零空间**。FCI 笛卡尔运动接口把该维暴露为 **elbow**（臂角），而不是让用户自己乘 $(I-J^+J)$。自己写力矩环时，官方示例仍显式构造零空间投影（与 Albu-Schäffer 2003 / Mayr 2024 同构）。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [Franka Research 3](../../wiki/entities/franka-research-3.md) | 硬件实体 |
| [零空间控制](../../wiki/concepts/null-space-control.md) | 公式与选型 |
| [Cartesian Impedance Controller](./cartesian-impedance-controller.md) | 机器人无关的开源实现 |
