# MAVSDK

> 来源归档

- **标题：** MAVSDK
- **类型：** repo
- **链接：** https://github.com/mavlink/MAVSDK
- **Stars：** ~0.9k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 面向 MAVLink 兼容飞控（PX4、ArduPilot）的 C++20 库与多语言绑定，提供高层异步 API（起飞、任务、Offboard、遥测订阅）。
- **沉淀到 wiki：** [mavsdk](../../wiki/entities/mavsdk.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**MAVSDK** 把底层 MAVLink 报文封装为 **插件式服务**（Action、Mission、Offboard、Telemetry、Param 等），适合伴机电脑、地面站原型与自动化测试，避免手写字节级协议。

支持 **C++、Python、Swift** 等；通过 UDP/TCP/串口连接 SITL 或真机。

---

## 典型用法

1. 连接 `udp://:14540`（PX4 SITL 默认）
2. `wait_until_ready` → `arm` → `takeoff`
3. `Offboard` 位置/速度/姿态设定点环（与 EGO-Planner 等规划器对接）
4. 订阅电池、GPS、姿态流用于日志或 RL 观测

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [px4_autopilot.md](px4_autopilot.md) | 主要目标飞控之一 |
| [ego_planner_swarm.md](ego_planner_swarm.md) | 规划输出常经 MAVROS/MAVSDK 转为 Offboard setpoint |
| [xtdrone.md](xtdrone.md) | 教学栈多用 MAVROS；MAVSDK 为更轻量的替代 API 层 |
