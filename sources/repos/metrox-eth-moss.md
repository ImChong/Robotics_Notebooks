# MOSS（metrox-eth/moss）

> 来源归档

- **标题：** MOSS — litter-picking rover
- **类型：** repo
- **来源：** Show Robotics / metrox-eth
- **链接：** <https://github.com/metrox-eth/moss>
- **入库日期：** 2026-10-06
- **核查日期：** 2026-10-06
- **一句话说明：** 开源履带式垃圾拾取移动操作机器人项目，仓库包含软件、固件、文档与硬件迭代记录。
- **项目页：** [Show Robotics · MOSS](../sites/showrobotics-moss.md)
- **关联仓库：** [MOSS × Jev 仿真](./metrox-eth-moss-jev.md)
- **沉淀到 wiki：** [MOSS 实体页](../../wiki/entities/moss.md)

## 仓库内容与状态

- **实机：** 首个原型已能行驶并携带机械臂；README 明确说明尚未自主完成垃圾拾取。
- **硬件：** 主要结构为履带底盘、手动清空的收纳箱、SO-101 衍生机械臂与 NormaCore 衍生夹爪；项目最新说明称 CAD 仍在实机验证和定版，STL / STEP 将随后发布。
- **计算与感知：** 台架配置为 Jetson Orin Nano Super 与 RealSense D455；另列 Pi 5 与不同相机的模块化方案，README 未将所有备选配置描述为已实机验证。
- **软件：** dimOS 用于遥操作、记录、导航与操作；ESP32-S3 控制电机。仓库开放软件与固件代码。
- **里程碑：** 2026-09-22 首次行驶；2026-09-26 机械臂与夹爪装机；2026-10-01 V0.5 CAD release candidate；2026-10-03 V0.6 模块化配置。
- **许可：** 软件 Apache-2.0；硬件 CERN-OHL-S-2.0；文档与媒体 CC BY 4.0。硬件许可不表示 CAD 文件已发布。

## 推荐入口

- 项目总览：[README.md](https://github.com/metrox-eth/moss/blob/main/README.md)
- 构建记录：[docs/build_log.md](https://github.com/metrox-eth/moss/blob/main/docs/build_log.md)
- 主仓库：<https://github.com/metrox-eth/moss>

## 对 wiki 的映射

- 项目页：[showrobotics-moss.md](../sites/showrobotics-moss.md)
- 仿真仓：[metrox-eth-moss-jev.md](./metrox-eth-moss-jev.md)
- 实体页：[moss.md](../../wiki/entities/moss.md)
