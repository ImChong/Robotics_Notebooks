# CyberBeast robot_mc：N32H473 裸机 FOC 关节固件 / Bare-Metal Joint FOC Firmware

> 来源归档（CyberBeast 官方 GitHub）

- **标题：** robot_mc
- **类型：** repo / firmware / joint-control
- **官方仓库：** https://github.com/cyberbeast0829/robot_mc
- **核对分支：** `CBA03_v4.2.89`
- **入库日期：** 2026-11-22
- **核对范围：** README 架构、构建、上位机部分；main.c 启动结构、foc.c 电流采样与 PWM、can_cyberbeast.c 协议定义与实时/主循环职责。一手入口：`README.md`、`Core/Src/main.c`、`Core/Src/foc.c`、`Core/Src/can_cyberbeast.c`；审查重点为本次软件栈定位，不声称覆盖仓库全部源码、历史提交或硬件实测。
- **通读：** 已完成
- **一句话说明：** 基于 Ben Katz mini-cheetah motorcontrol 移植的 N32H473CE 三相 FOC 固件，40 kHz 中断控制、CAN/CAN-FD 参数访问与双页 TLV 持久化。
- **开源状态：** 已公开源码；对应构建与真机兼容性以仓库 README 和具体硬件版本为准。
- **沉淀到 wiki：** 是 → [cyberbeast-robot-mc](../../wiki/entities/cyberbeast-robot-mc.md)

## 一手事实与代码定位

ATIM1 触发 ADC 注入采样，SPI1 + DMA2 读取编码器，控制中断运行 FOC 和 CYBERBEAST 控制帧处理；JSON endpoint 描述符流由主循环发送；配置用带 CRC32 的双页 TLV 持久化。

## 官方构建 / 运行入口

- `make`
- `make check`
- `make hosttest`
- `python tools/mc_ep.py --can tree`

## 工程边界

40 kHz 是固件内部控制环频率，不等于主站可达到 40 kHz 的总线指令刷新率；量程、引脚与标定参数以具体板卡和固件版本为准。

## 参考原文

- [README.md](https://github.com/cyberbeast0829/robot_mc/blob/CBA03_v4.2.89/README.md)
- [Core/Src/main.c](https://github.com/cyberbeast0829/robot_mc/blob/CBA03_v4.2.89/Core/Src/main.c)
- [Core/Src/foc.c](https://github.com/cyberbeast0829/robot_mc/blob/CBA03_v4.2.89/Core/Src/foc.c)
- [Core/Src/can_cyberbeast.c](https://github.com/cyberbeast0829/robot_mc/blob/CBA03_v4.2.89/Core/Src/can_cyberbeast.c)
