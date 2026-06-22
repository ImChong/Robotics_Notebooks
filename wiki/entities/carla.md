---
type: entity
tags: [entity, simulator, autonomous-driving, carla, urban, sensor-simulation]
status: complete
updated: 2026-06-22
related:
  - ./airsim.md
  - ../concepts/sim2real.md
  - ../overview/sim-platforms-decade-technology-map.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "面向自动驾驶的开源城市仿真平台：Unreal Engine 驱动的高保真城市场景、多传感器套件与交通参与者，是 AD 感知–规划–控制闭环与 Sim2Real 研究的经典基础设施。"
---

# CARLA

**CARLA**（Car Learning to Act）是面向 **自动驾驶** 研究的开源 **城市驾驶仿真器**，以 Unreal Engine 提供高保真视觉与物理交互环境。

## 一句话定义

> CARLA 为自动驾驶提供 **可编程城市沙盒**：动态交通、多相机/激光雷达传感器与可重复场景，让感知、预测、规划算法在仿真中安全迭代后再上路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AD | Autonomous Driving | 自动驾驶 |
| LiDAR | Light Detection and Ranging | 激光雷达 |
| Sim2Real | Simulation to Real | 仿真到真实部署迁移 |
| API | Application Programming Interface | 仿真控制与传感器接口 |

## 为什么重要

虽非人形具身主线，CARLA 在 **移动机器人 Sim2Real** 方法论上影响深远：

1. **传感器丰富**：RGB、深度、语义分割、LiDAR、IMU 等标准套件。
2. **交通生态**：NPC 车辆/行人、天气与光照随机化，支撑 **域随机化** 研究。
3. **社区基准**：大量 AD 数据集与 leaderboard 以 CARLA 为后端或对照。

深蓝具身智能 [十年盘点](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md) 在文尾将其与具身平台并列，强调 **各垂直方向的基础设施价值**。

## 核心结构/机制

- **UE 渲染**：城市地图、建筑与动态对象。
- **Python API**：同步/异步模式控制 ego 车辆与传感器流。
- **Scenario Runner**：可复现交通场景脚本。

## 常见误区或局限

- **误区：CARLA = 人形仿真** — 本体是 **轮式车辆**；足式/人形见 [Isaac Lab](./isaac-lab.md)、[MuJoCo](./mujoco.md)。
- **局限：操作与室内** — 室内交互见 [AI2-THOR](./ai2-thor.md)、[Habitat](./habitat-sim.md)。

## 关联页面

- [AirSim](./airsim.md) — 另一 UE 系机器人视觉仿真
- [Sim2Real](../concepts/sim2real.md)
- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- Dosovitskiy et al., *CARLA: An Open Urban Driving Simulator* — [arXiv](https://arxiv.org/abs/1711.03938)

## 推荐继续阅读

- [CARLA 官方文档](https://carla.org/)
- [仿真器选型指南](../queries/simulator-selection-guide.md)
