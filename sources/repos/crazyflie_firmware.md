# crazyflie-firmware

> 来源归档

- **标题：** Crazyflie Firmware
- **类型：** repo
- **链接：** https://github.com/bitcraze/crazyflie-firmware
- **Stars：** ~1.5k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** Bitcraze Crazyflie 2.x / Bolt / Roadrunner 的嵌入式飞控固件：STM32 上姿态控制、CRTP 无线协议、扩展甲板与定位支持。
- **沉淀到 wiki：** [crazyflie-firmware](../../wiki/entities/crazyflie-firmware.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

与 **PX4** 不同，Crazyflie 走 **轻量级微四轴** 路线：机载算力有限，强调 **CRTP**（Crazy RealTime Protocol）与 Python 上位机（cfclient、cflib）。适合 **群体飞行、动捕、Swarm 研究**。

固件含：传感器融合、PID 控制、参数系统、日志、UWB/Lighthouse/OptiTrack 定位模块接口。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [crazyswarm2.md](crazyswarm2.md) | 多机编排依赖本固件 + cflib |
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | 仿真环境提供 Crazyflie 动力学模型选项 |
| [flightmare.md](flightmare.md) | 高保真渲染仿真可对照真机 Crazyflie |
