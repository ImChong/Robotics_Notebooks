# gym-pybullet-drones

> 来源归档

- **标题：** gym-pybullet-drones
- **类型：** repo
- **链接：** https://github.com/utiasDSL/gym-pybullet-drones
- **Stars：** ~2.0k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** **PyBullet + Gymnasium** 的单/多智能体四旋翼 RL 环境：CF2X、HB 等模型，内置 PID/RL 控制接口，论文引用广泛的无人机 RL 基准。
- **沉淀到 wiki：** [multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

由 **UTIAS DSL** 维护，把四旋翼动力学封装为 **标准 Gym API**：

- `CtrlAviary` / `VisionAviary` 等环境变体
- 动作空间：RPM、PID 目标、one-step RL 动作
- 支持 **多机** 交互与对抗/编队任务原型
- 可选 **Betaflight** 风格参数、Crazyflie 尺度

适合 **快速验证 RL、MARL、控制课程作业**；物理保真度低于 MuJoCo/Flightmare，但依赖轻、可复现性强。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [quad_swarm_rl.md](quad_swarm_rl.md) | 同为 OpenAI Gym 风格 swarm 环境，可对照 API |
| [mujoco](../../wiki/entities/mujoco.md) | 腿式/臂式 RL 常用 MuJoCo；本仓为 **空中** PyBullet 专精 |
| [px4_autopilot.md](px4_autopilot.md) | Sim2Real 需另接 SITL；本环境不直接跑 PX4 |
