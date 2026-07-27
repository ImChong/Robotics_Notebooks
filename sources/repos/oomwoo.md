# makerspet/oomwoo

> 来源归档

- **标题：** OOMWOO
- **类型：** repo（开源家用扫地机器人总仓 / 模块贡献枢纽）
- **组织：** Maker's Pet（[makerspet](https://github.com/makerspet)）
- **链接：** https://github.com/makerspet/oomwoo
- **项目页：** https://oomwoo.com/
- **教程：** https://makerspet.com/learn/ · [Gazebo 仿真教程](https://makerspet.com/blog/simulate-oomwoo-one-robot-vacuum-in-gazebo-with-ros-2/)
- **许可：** Apache-2.0
- **星标（截至 2026-07-27）：** ~6579
- **Topics：** `ros2` · `slam` · `lidar` · `raspberry-pi` · `home-assistant` · `3d-printing` · `vacuum-robot`
- **入库日期：** 2026-07-27
- **一句话说明：** 可自建的开源家用扫地机器人：ROS 2 / Nav2 / 2D LiDAR、树莓派 CM4·CM5、3D 打印底盘、本地优先（Home Assistant），仿真优先、模块并行贡献。
- **开源状态：** **部分开源 / 早期开发** — 架构与接口文档、Gazebo URDF、安装环境已公开；完整 BoM、CAD、I/O PCB 与固件仍在推进。
- **项目页归档：** [oomwoo-com.md](../sites/oomwoo-com.md)
- **沉淀到 wiki：** [oomwoo](../../wiki/entities/oomwoo.md)

---

## 为什么值得保留

- 把 **消费级扫地机形态** 接到 **ROS 2 + Nav2 + slam_toolbox** 教学/DIY 主线，是室内 AMR 栈的可动手整机入口。
- **CPU/MCU 安全分层**（Linux/ROS2 不负责硬安全）与 **仿真优先、接口契约** 的社区并行开发模式，对开源整机工程有参考价值。
- 与 [Navigation2](../../wiki/entities/navigation2.md)、[SLAM Toolbox](../../wiki/entities/slam-toolbox.md)、[导航·SLAM 栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md) 直接对齐。

## 配套仓库（同组织，截至 2026-07-27）

| 仓库 | 角色 | 状态摘录 |
|------|------|----------|
| [makerspet/oomwoo](https://github.com/makerspet/oomwoo) | 总仓：架构、模块 RFC、贡献入口 | early development；~6.5k★ |
| [makerspet/oomwoo-one](https://github.com/makerspet/oomwoo-one) | ROS 2 robot description / Gazebo 仿真 | 已开源（占位 URDF） |
| [makerspet/oomwoo-install](https://github.com/makerspet/oomwoo-install) | ROS 2 / Ubuntu 安装与 Docker 环境 | 已开源 |
| [makerspet/oomwoo-one-cad](https://github.com/makerspet/oomwoo-one-cad) | 3D CAD | 进行中 |
| [makerspet/oomwoo-io-board](https://github.com/makerspet/oomwoo-io-board) | I/O + 电机驱动 PCB（KiCad） | 进行中 |
| [makerspet/oomwoo-io-firmware](https://github.com/makerspet/oomwoo-io-firmware) | STM32 MCU 固件 | 进行中 |
| [makerspet/proscenic-m6pro](https://github.com/makerspet/proscenic-m6pro) | 占位真机（Proscenic M6 Pro）ROS 2 描述 | 临时桥接，待自研硬件落地 |

## 架构要点（摘自 ARCHITECTURE.md）

| 层 | 选型 / 约定 |
|----|-------------|
| 计算（CPU） | Raspberry Pi **CM4 / CM5**（或 pin 兼容模块）；跑 ROS 2 · slam_toolbox · Nav2 · 高层行为 |
| 教育变体 | 同载板上插 **ESP32-S3**（CM 外形）跑 micro-ROS；SLAM/Nav **离机** 到开发 PC |
| 实时 / 安全（MCU） | 暂定 **STM32G070** 类高 GPIO MCU + FreeRTOS；电机、编码器、保险杠/悬崖/轮跌落、充电；**硬安全不依赖 Linux** |
| CPU↔MCU | **自定义高速串口**（非 micro-ROS）+ 健康包 / CPU reset GPIO |
| 传感 | 2D LiDAR（UART ~5 Hz，兼容 `kaiaai/LDS`）挂 CPU；bumper/cliff 挂 MCU；近场避障（相机 + ToF）为后期模块 |
| 仿真 | Gazebo + URDF；住宅布局世界；**先仿真后真机** |
| MVP（目标 2026-08-31） | CM 类算力 · LiDAR · 手动 SLAM · teleop · 3D 打印底盘 · Gazebo · 演示；**不含** 自动覆盖清扫、回充、Home Assistant、应用层 |
| 成本叙事 | 外购件约 **~$200** + Pi；对标中端商用机能力 |

## 软件模块（贡献面）

仿真可先于硬件：`urdf-gazebo-sim`、`clean-and-map`、`nav-localize`、`dock-cycle`、`recovery-safety`、`health-monitor`、`obstacle-avoidance`、`cleaning-jobs`、`control-app` 等；公共话题契约见仓内 `docs/SOFTWARE_INTERFACES.md`（`/scan`、`/odom`、`/cmd_vel`、`/map`、标准 TF）。

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 主仓 | **已开源**（Apache-2.0）：架构、贡献模块表、BoM 草稿、接口草案 |
| 仿真 / 安装 | **已开源**：`oomwoo-one`、`oomwoo-install` |
| 硬件交付物 | **部分 / 进行中**：BoM、CAD、I/O 板与固件仓库存在但未完成可复现整机 BOM |
| 项目页 | [oomwoo.com](https://oomwoo.com/) 与主仓互指；无「将开源」空承诺——代码已在 GitHub |

## 对 wiki 的映射

- [OOMWOO](../../wiki/entities/oomwoo.md)
- [Navigation2](../../wiki/entities/navigation2.md)
- [SLAM Toolbox](../../wiki/entities/slam-toolbox.md)
- [导航·SLAM·自动驾驶栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md)
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md)
- [MuSHR](../../wiki/entities/mushr.md)（同属低成本 ROS 移动平台对照）
