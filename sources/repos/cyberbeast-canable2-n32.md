# CyberBeast canable2-fw-n32：N32H473 USB-CAN 桥接固件 / USB-to-CAN Firmware

> 来源归档（CyberBeast 官方 GitHub）

- **标题：** canable2-fw-n32
- **类型：** repo / firmware / joint-control
- **官方仓库：** https://github.com/cyberbeast0829/canable2-fw-n32
- **核对分支：** `main`
- **入库日期：** 2026-11-22
- **核对范围：** README SLCAN 命令与构建；PORTING_PLAN 芯片移植边界；src/main.c USB/CAN 轮询主循环；src/slcan.c CAN 与 ASCII 转码部分。一手入口：`README.md`、`PORTING_PLAN.md`、`src/main.c`、`src/slcan.c`；审查重点为本次软件栈定位，不声称覆盖仓库全部源码、历史提交或硬件实测。
- **通读：** 已完成
- **一句话说明：** 将 CANable2 SLCAN 固件移植到 N32H473 的 USB CDC-to-CAN/CAN-FD 桥接程序，支持标准及扩展帧 ASCII 命令。
- **开源状态：** 已公开源码；对应构建与真机兼容性以仓库 README 和具体硬件版本为准。
- **沉淀到 wiki：** 是 → [cyberbeast-canable2-n32](../../wiki/entities/cyberbeast-canable2-n32.md)

## 一手事实与代码定位

固件 main 初始化 system/can/led/usb，在循环中执行 led_process、can_process、cdc_process，再将 FDCAN RX 帧转为 ASCII 回送 USB CDC；slcan.c 解析 `O`、`S6`、`t/T` 与 CAN FD 的 `d/D/b/B` 等命令。

## 官方构建 / 运行入口

- `make`
- `将设备接为 USB CDC 后打开 SLCAN 串口`
- `S6 (500k) → O (打开) → t/T/d/D/b/B 发送帧`

## 工程边界

本页说明 main 分支 SLCAN 实现；PORTING_PLAN 中部分引脚是规划建议而非最终硬件定版，必须按实际 PCB 校核；USB CDC/ASCII 具有额外时延，不可自动等同硬实时 SocketCAN。

## 参考原文

- [README.md](https://github.com/cyberbeast0829/canable2-fw-n32/blob/main/README.md)
- [PORTING_PLAN.md](https://github.com/cyberbeast0829/canable2-fw-n32/blob/main/PORTING_PLAN.md)
- [src/main.c](https://github.com/cyberbeast0829/canable2-fw-n32/blob/main/src/main.c)
- [src/slcan.c](https://github.com/cyberbeast0829/canable2-fw-n32/blob/main/src/slcan.c)
