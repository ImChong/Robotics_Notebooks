# drift-mpc-ackermann

> 来源归档

- **标题：** Friction-Adaptive MPC for Autonomous Drift on Ackermann Vehicles
- **类型：** repo
- **来源：** Gelminaio（GitHub）
- **链接：** https://github.com/Gelminaio/drift-mpc-ackermann
- **Stars：** ~4（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** 1:10 Ackermann 低摩擦面漂移：非线性 MPC（Pacejka 轮胎 + 自行车模型）+ 在线摩擦估计；ROS 2 Jazzy 分布式栈（ESP32 / RPi4 / 主机），含 Sim2Real 随机摩擦仿真管线。
- **代码：** https://github.com/Gelminaio/drift-mpc-ackermann（**已开源**）
- **沉淀到 wiki：** 景观页

---

## 核心定位

- **控制：** 饱和摩擦区轨迹跟踪（主动利用侧滑而非回避）
- **估计：** IMU + 轮速里程计在线估计 μ，自适应 MPC 模型与约束
- **硬件：** RPLIDAR A1、BNO085、ESP32 实时层

---

## 关联

- MPC：[`wiki/methods/model-predictive-control.md`](../../wiki/methods/model-predictive-control.md)
- RL 漂移对照：[`xcar_rlgpu.md`](./xcar_rlgpu.md)、[`gym_khana.md`](./gym_khana.md)
