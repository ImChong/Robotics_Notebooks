---
type: entity
tags: [repo, firmware, crazyflie, micro-quadrotor, embedded, bitcraze]
status: complete
updated: 2026-05-27
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./crazyswarm2.md
  - ./px4-autopilot.md
  - ./gym-pybullet-drones.md
sources:
  - ../../sources/repos/crazyflie_firmware.md
summary: "crazyflie-firmware 是 Bitcraze Crazyflie 2.x / Bolt 的机载飞控固件：STM32 姿态控制、CRTP 协议与扩展甲板（UWB、Lighthouse、OptiTrack）支持。"
---

# Crazyflie Firmware

**crazyflie-firmware**（[bitcraze/crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware)）是 **Bitcraze** 微四轴平台的 **机载实时固件**，与 [PX4](./px4-autopilot.md) 同属「飞控层」但面向 **27g 级微机体** 与 **CRTP** 生态。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |

## 为什么重要

- **群体飞行硬件底座**：[Crazyswarm2](./crazyswarm2.md) 依赖本固件 + cflib 上位机。
- **教学与研究友好**：成本低、风险可控，适合控制与 swarm 算法入门。
- [gym-pybullet-drones](./gym-pybullet-drones.md) 提供 **Crazyflie 尺度** 动力学选项，可与真机对照。

## 核心结构/机制

| 模块 | 说明 |
|------|------|
| **传感器融合** | IMU、气压计；外接定位甲板 |
| **控制** | 姿态/速率 PID，电机混控 |
| **CRTP** | 与 PC/机载计算机的轻量协议 |
| **Deck 驱动** | Lighthouse、Loco、OptiTrack 等扩展 |
| **参数/日志** | 在线调参与飞行日志 |

上位机常用 **cfclient**、**cflib**（Python）发送设定点或高层命令。

## 常见误区或局限

- **误区：Crazyflie = PX4 小飞机** — 协议、算力、任务模式完全不同，勿混用工具链。
- **局限：室外长航程** — 微型机体受风与续航限制大。
- **局限：重载操作** — 不适合工业级载荷与人机协作臂。

## 参考来源

- [sources/repos/crazyflie_firmware.md](../../sources/repos/crazyflie_firmware.md)
- [bitcraze/crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware)

## 关联页面

- [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [Crazyswarm2](./crazyswarm2.md)
- [PX4 Autopilot](./px4-autopilot.md)（标准多旋翼飞控对照）

## 推荐继续阅读

- [Bitcraze 官方文档](https://www.bitcraze.io/documentation/)
