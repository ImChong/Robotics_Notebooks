# Blue Core Software（berkeleyopenarms/blue_core）

> 来源归档

- **标题：** Blue Core Software
- **类型：** repo
- **来源：** Berkeley Open Arms / UC Berkeley
- **链接：** <https://github.com/berkeleyopenarms/blue_core>
- **项目页：** <https://berkeleyopenarms.github.io/>
- **论文：** <https://arxiv.org/abs/1904.03815>
- **许可：** MIT（`blue_core`；同组织其他辅助仓库需逐项核查）
- **入库日期：** 2026-07-28
- **一句话说明：** Blue 机械臂的 ROS 核心运行栈，含 bringup、控制器管理、硬件抽象、底层驱动、URDF 与消息定义。
- **开源状态：** **部分开源（系统级）**；核心控制软件可运行，完整硬件制造链与所有辅助资产不由单一 MIT 仓库覆盖。
- **沉淀到 wiki：** [QDD / Blue 论文实体](../../wiki/entities/paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti.md)

## 仓库结构与入口

| 包 / 入口 | 作用 |
|-----------|------|
| `blue_bringup` | launch、配置和启动脚本 |
| `blue_controller_manager` | 基于 `ros_control` 动态切换控制器 |
| `blue_hardware_interface` | 关节消息与执行器消息之间的抽象 |
| `blue_controllers` | 自定义控制器插件 |
| `blue_hardware_drivers` | 电机驱动通信与 ROS 接口 |
| `blue_descriptions` | URDF 和三维模型 |

典型启动：准备 `blue_configs` 参数文件后执行 `roslaunch blue_bringup right.launch param_file:=blue_params.yaml`；左臂使用对应 `left.launch`。

## 对 wiki 的映射

- 项目页：[berkeley-open-arms-blue.md](../sites/berkeley-open-arms-blue.md)
- 论文归档：[humanoid_pnb_quasi-direct-drive-for-low-cost-compliant-roboti.md](../papers/humanoid_pnb_quasi-direct-drive-for-low-cost-compliant-roboti.md)
