# Betaflight

> 来源归档

- **标题：** Betaflight
- **类型：** repo（飞控固件）
- **来源：** Betaflight 开源社区
- **链接：** https://github.com/betaflight/betaflight
- **官网：** https://betaflight.com/
- **Stars：** ~11.1k（2026-06）
- **入库日期：** 2026-06-17
- **许可证：** GPLv3
- **一句话说明：** 面向 **FPV 竞速 / 自由式 / 特技** 多旋翼（及固定翼）的 **开源飞控固件**：极致姿态环性能、DShot/Oneshot 电调协议、Blackbox 日志、内置 OSD、MSP 串口生态；与 PX4 的 **自主导航 / MAVLink** 路线不同。
- **沉淀到 wiki：** [betaflight](../../wiki/entities/betaflight.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位（README）

- **赛道**：手飞 FPV、竞速、自由式；优化 **摇杆跟踪、扰动响应、信号处理延迟**，而非 Offboard 自主任务。
- **配置**： [Betaflight App](https://app.betaflight.com)（PWA）；旧称 Betaflight Configurator 能力已并入 App。
- **版本**：自 4.x 起迁移至 **`YYYY.M.PATCH`**（如 `2025.12.x`、`2026.6.x`），每年 **6 月 / 12 月** 大版本；Alpha → Beta（特性冻结）→ RC → Release。
- **硬件**：支持 STM32 **F4 / G4 / F7 / H7** 等目标板；团队 **不制造硬件**，问题联系板卡厂商。

---

## 主要特性（README 摘录）

| 类别 | 能力 |
|------|------|
| **电机协议** | DShot（150/300/600）、Multishot、Oneshot（125/42）、Proshot1000 |
| **接收机** | PWM、PPM、SPI、Serial（SBus、SumH、SumD、Spektrum、XBus 等）+ failsafe |
| **遥测** | CRSF、FrSky、HoTT smart-port、**MSP** 等 |
| **日志** | **Blackbox** 至板载 Flash 或外接 microSD |
| **OSD** | 无需第三方 OSD 固件即可配置 |
| **调参** | 飞行中 PID / rate 调整；滑块式 PID 与滤波器 tuning；多 rate profile |
| **外设** | GPS、VTX（Unify Pro / IRC Tramp）、OLED、WS2811 RGB 灯带等 |
| **端口** | 可配置串口映射（RX、遥测、ESC telemetry、MSP、GPS、OSD…） |

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [betaflight-com.md](../sites/betaflight-com.md) | 官方站点、文档与 Partner 计划 |
| [px4_autopilot.md](px4_autopilot.md) | 研究/工业 **自动驾驶仪**；MAVLink + SITL + Offboard |
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | RL 仿真可选 **Betaflight 风格** 动力学/控制参数 |
| [crazyflie_firmware.md](crazyflie_firmware.md) | 微四轴 CRTP 栈；算力与用途均不同 |
| [multirotor_uav_stack_catalog.md](multirotor_uav_stack_catalog.md) | 十仓多旋翼索引；本仓补 **FPV 飞控** 维度 |

---

## 对 wiki 的映射

- 新建实体页 [`wiki/entities/betaflight.md`](../../wiki/entities/betaflight.md)：FPV 飞控固件选型、与 PX4 分层对照、MSP/Blackbox 工具链。
- 更新 [`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md)：飞控层增加 **FPV / 手飞** 分支说明。
