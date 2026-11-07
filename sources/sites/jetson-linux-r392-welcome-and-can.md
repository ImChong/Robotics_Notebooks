# Jetson Linux r39.2 开发指南：Welcome 与 CAN 章节

> 来源归档（site / 官方文档节选；整本开发指南体量很大，本档只覆盖下列两页，整本指南见 [jetson-linux-r392-developer-guide.md](./jetson-linux-r392-developer-guide.md)）

- **标题：** NVIDIA Jetson Linux Developer Guide 39.2 — Welcome；Controller Area Network (CAN)
- **类型：** site / 官方技术文档
- **链接：** <https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/>（Welcome）、<https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ControllerAreaNetworkCan.html>（CAN）
- **版本：** Jetson Linux **39.2 GA**，支持 Jetson Thor 与 Jetson Orin 两个系列
- **入库日期：** 2026-11-07
- **核对范围：** Welcome 页全文（含设备/P-number 对照表、DevKit 与量产模组区别、SDK Manager / Quick Start / Module Adaptation 指引）与 CAN 章节全文（平台差异表、启用步骤、DTB、pinmux、内核模块、SocketCAN 配置、can-utils 用法、环回调试、TDCR 高速率、CAN 时钟）。**不覆盖**开发指南的其余章节（引导、内核定制、电源与性能、相机、安全、OTA 等）。
- **通读：** 已完成
- **沉淀到 wiki：** [NVIDIA Jetson](../../wiki/entities/nvidia-jetson.md)、[CAN 总线](../../wiki/concepts/can-bus-protocol.md)

## 核对摘录

### Welcome

- 本指南适用于 Jetson Linux 39.2 GA，覆盖 **Jetson Thor** 与 **Jetson Orin**（AGX Orin / Orin NX / Orin Nano）。JetPack 捆绑 Jetson 平台软件；Jetson Linux 提供 Linux 内核、引导程序、NVIDIA 驱动、刷机工具、示例文件系统等。
- **开发者套件 vs 模组：** 开发者套件是「非量产规格模组 + 参考载板」，**不用于量产**；量产模组单独销售、出厂**不预装软件**，须装到自己设计或采购的载板并刷入自己开发的镜像。用 NVIDIA SDK Manager 安装 Jetson Linux 与 JetPack 组件；也可按 Quick Start 只刷引导程序与文件系统；从开发套件迁移到量产模组与载板见「Jetson Module Adaptation and Bring-Up」。
- 型号对照：Thor 系列 T5000 128GB（P3834-0008，仅开发套件）/ T4000 64GB（P3834-0000，量产），AGX Thor 开发套件 P4070；Orin NX 16GB（P3767-0000）/ 8GB（P3767-0001）；Orin Nano 8GB（P3767-0003）/ 4GB（P3767-0004）；AGX Orin 32GB（P3701-0004）/ 64GB（P3701-0005）/ Industrial（P3701-0008）等。

### CAN 章节

- 适用平台：Jetson Thor、AGX Orin、Orin NX、Orin Nano。片上是 **Time Triggered CAN（TTCAN）控制器**（驱动 `mttcan`），支持标准帧与扩展帧，位速率可配 10 kbps–1 Mbps；**数据位速率 Thor 最高 8 Mbps，其他平台最高 15 Mbps**。
- 控制器数量：Orin NX / Orin Nano 1 个，AGX Orin 2 个，Thor 2 个；Thor 开发套件自带收发器（载板 J47 头引出 FSI_CAN0/1 H/L，并有端接与环回短接说明），其余平台需外接支持 ≥3.3 V 的收发器（NVIDIA 推荐 WaveShare SN65HVD230 板用于开发）。
- 启用：内核设备树里 `mttcan` 节点默认 `status = "okay"`；Thor 默认引脚即 CAN 功能，Orin 系列需按表设置 pinmux（`busybox devmem` 临时写寄存器，或改 pinmux 配置并重新刷机 / 用 Jetson-IO）；依序 `modprobe can`、`can_raw`、`mttcan`。
- 配置：`ip link set can0 up type can bitrate 500000 dbitrate 1000000 berr-reporting on fd on`（FD 模式示例，仲裁段 500 kbps、数据段 1 Mbps）；`can-utils` 的 `cansend` / `candump` / `cangen` 收发测试；FD 帧用 `123##1abcdabcd` 语法。
- 调试：`ip link set can0 type can bitrate 500000 loopback on` 做内部环回确认控制器工作；仍收发失败则查收发器连线、示波器；出现「No buffer space available」可用 `cangen -L 8 can0 -p 1000` 的轮询方式。
- 更高速率：Orin 系列通过 sysfs `tdcr`（Transmission Delay Compensation Register）调节（需收发器支持）；CAN 时钟：Orin 系列默认 50 MHz，Thor 固定 40 MHz。
