# Jetson Linux Developer Guide（r39.2）

> 来源归档

- **标题：** NVIDIA Jetson Linux Developer Guide — Welcome
- **类型：** site（官方 BSP/定制文档）
- **链接：** <https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/>
- **对应版本：** **NVIDIA Jetson Linux 39.2 GA**（JetPack 7 系）
- **入库日期：** 2026-09-06
- **一句话说明：** Jetson **BSP 定制主文档**：DevKit vs 量产模组、SDK Manager 刷机、Module Adaptation and Bring-Up、Thor/Orin 全系列 P-number 对照。
- **沉淀到 wiki：** [`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **文档** | 公开开发者指南 |
| **Jetson Linux 源码/二进制** | 随 **JetPack** 分发；非独立 GitHub 一键 clone 全栈 |
| **定制** | 载板适配、内核/设备树、安全、OTA — 按指南在 BSP 工作区修改 |

## Welcome 页要点（2026-09-06）

### 版本与产品线

- **Jetson Linux 39.2 GA** 同时支持 **Jetson Thor** 与 **Jetson Orin** 家族
- **JetPack** 捆绑 Jetson 平台软件；Jetson Linux 提供 **Linux kernel、bootloader、NVIDIA 驱动、刷机工具、sample rootfs** 等

### DevKit vs 量产模组

| 类型 | 用途 |
|------|------|
| **Developer Kit** | 非量产规格模组 + 参考载板；用 JetPack **开发与测试** |
| **Commercial Module** | 量产部署；**无预装软件**；自研/采购载板后刷入开发阶段镜像 |

量产迁移：见 **Jetson Module Adaptation and Bring-Up**（按模组分册）。

### 刷机与工具

- **SDK Manager** — 在 DevKit 上安装 Jetson Linux 与其余 JetPack 组件（推荐）
- **Quick Start** — 仅刷 bootloader + rootfs（不含全部 JetPack 组件）

### 本文档覆盖设备（节选）

| 系列 | 模组示例 | DevKit |
|------|----------|--------|
| **Jetson Thor** | T5000 128GB (P3834-0008)；T4000 64GB | AGX Thor DevKit (P4070) |
| **AGX Orin** | 32GB/64GB/Industrial | AGX Orin DevKit (P3730) |
| **Orin NX** | 8GB/16GB | Orin Nano DevKit (P3766) |
| **Orin Nano** | 4GB/8GB | Orin Nano DevKit (P3766) |

文档标题/小节会标注 **Applies to … only** 以区分代际差异。

### 相关入口

- Jetson Download Center · Jetson FAQ · Autonomous Machines Getting Started
- 与 [`nvidia-jetpack.md`](./nvidia-jetpack.md)、[`jetson-ai-lab.md`](./jetson-ai-lab.md) 分工：本指南 = **BSP/刷机/定制**；AI Lab = **模型/应用教程**

## 对 wiki 的映射

- 平台：[`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)
- Orin NX 深读：[`wiki/entities/jetson-orin-nx.md`](../../wiki/entities/jetson-orin-nx.md)
- JetPack：[`sources/sites/nvidia-jetpack.md`](./nvidia-jetpack.md)
- HIL：[`wiki/concepts/hardware-in-the-loop.md`](../../wiki/concepts/hardware-in-the-loop.md)
