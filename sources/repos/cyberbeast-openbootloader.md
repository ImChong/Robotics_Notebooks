# CyberBeast OpenBootloader：N32H473 CAN OTA 引导程序 / CAN Bootloader

> 来源归档（CyberBeast 官方 GitHub）

- **标题：** OpenBootloader
- **类型：** repo / firmware / joint-control
- **官方仓库：** https://github.com/cyberbeast0829/OpenBootloader
- **核对分支：** `LION_H473_PB67`
- **入库日期：** 2026-11-22
- **核对范围：** README 目标硬件、CAN 协议、启动与升级章节；Core/Src/main.c 启动跳转代码；app_openbootloader.c 的 FDCAN 接口与内存注册。一手入口：`README.md`、`Core/Src/main.c`、`OpenBootloader/App/app_openbootloader.c`、`CAN_Bootloader_Protocol.md`；审查重点为本次软件栈定位，不声称覆盖仓库全部源码、历史提交或硬件实测。
- **通读：** 已完成
- **一句话说明：** 面向 N32H473CE 的 Classic CAN IAP 引导程序，Bootloader 从 0x08000000 占用 32 KB，500 kbit/s 升级 APP。
- **开源状态：** 已公开源码；对应构建与真机兼容性以仓库 README 和具体硬件版本为准。
- **沉淀到 wiki：** 是 → [cyberbeast-openbootloader](../../wiki/entities/cyberbeast-openbootloader.md)

## 一手事实与代码定位

上电配置后轮询握手 200 ms，无主站时跳转至 APP 0x08008000；APP 可通过共享 RAM 0x20023FF0 写 0xB00710AD 并软复位要求进入引导；仅注册 FDCAN 而非 USART 通道。

## 官方构建 / 运行入口

- `Keil MDK-ARM 构建 N32H473 target`
- `python tools/blcan.py sync`
- `python tools/blcan.py upgrade --firmware <bin> --addr 0x08008000 --verify --go-addr 0x08008000`

## 工程边界

这条升级链使用 Classic CAN 500 kbit/s、11 位 ID，不能直接照搬应用固件的 29 位扩展帧或 CAN FD 参数；擦写和起跳须核对地址并保留配置区域。

## 参考原文

- [README.md](https://github.com/cyberbeast0829/OpenBootloader/blob/LION_H473_PB67/README.md)
- [Core/Src/main.c](https://github.com/cyberbeast0829/OpenBootloader/blob/LION_H473_PB67/Core/Src/main.c)
- [OpenBootloader/App/app_openbootloader.c](https://github.com/cyberbeast0829/OpenBootloader/blob/LION_H473_PB67/OpenBootloader/App/app_openbootloader.c)
- [CAN_Bootloader_Protocol.md](https://github.com/cyberbeast0829/OpenBootloader/blob/LION_H473_PB67/CAN_Bootloader_Protocol.md)
