---
type: entity
tags: [mobile-robot, education, ros, autonomous-driving, open-source]
status: complete
updated: 2026-10-27
project_id: duckietown-duckiebot
code: https://github.com/duckietown/dt-duckiebot-interface
summary: "Duckiebot：Duckietown 生态的标准化开源小车，Raspberry Pi + ROS，配 gym-duckietown 仿真与真实道路系统，用于自主驾驶教育与研究。"
related:
  - ./turtlebot3.md
  - ./paper-openbot-smartphone-mobile-robot.md
  - ../concepts/sim2real.md
  - ./paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# Duckiebot（Duckietown）

## 一句话定义

**Duckiebot** 是 Duckietown 生态中的标准化开源小车，组织 [duckietown](https://github.com/duckietown)，关键仓库 [dt-duckiebot-interface](https://github.com/duckietown/dt-duckiebot-interface)、[dt-core](https://github.com/duckietown/dt-core)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS | Robot Operating System | 机器人中间件 |
| Gym | OpenAI Gym interface | 强化学习环境接口 |
| MARL | Multi-Agent RL | 多智能体强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- 自动驾驶教学领域的标准化平台；本库另有 Duckietown 论文清单条目，本页是平台节点。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **18** 位。

## 核心要点（据文章）

- 基于 Raspberry Pi，支持 ROS，有完整的仿真环境（gym-duckietown）和真实道路系统。
- 代码高度模块化，组织下有数百个仓库；适合学习感知、规划和多智能体。

## 机制与架构图

```mermaid
flowchart LR
    Core["dt-core (感知 / 规划 / 多智能体 MARL 模块)"] --> Sim["gym-duckietown 仿真环境调试"]
    Sim -->|"Sim2Real 迁移"| Interface["dt-duckiebot-interface (ROS 中间件接口)"]
    Interface --> HW["Raspberry Pi 机载计算"]
    HW --> Real["Duckiebot 真车与真实道路系统"]
```

## 源码运行时序图

在底层硬件接口仓 [`duckietown/dt-duckiebot-interface`](https://github.com/duckietown/dt-duckiebot-interface)（`ente` 分支）中，容器启动脚本与 `duckiebot_interface` 启动器按机器人硬件型号拉起基于 DTPS（HTTP / 可选共享内存 SHM）的传感器与执行器驱动节点：

```mermaid
sequenceDiagram
    autonumber
    participant Boot as launchers/*.sh<br/>duckiebot_interface/launch/main.py
    participant Cam as camera_driver_node.*<br/>CameraNodeAbs
    participant DTPS as DTPS HTTP / SHM<br/>DT_CAMERA_SHM_OUT_PATH
    participant Whl as wheels_driver_node.main<br/>WheelsDriverNode
    participant Drv as DaguWheelsDriver<br/>VirtualWheelsDriver

    Boot->>Boot: 读取 ROBOT_TYPE / ROBOT_HARDWARE / ROBOT_CONFIGURATION
    Boot->>Cam: 启动 camera_driver_node.<robot_hardware>
    Boot->>Whl: 启动 wheels_driver_node.main（--actuator-name base）
    Cam->>Cam: 采集图像并经 HardwareJpegEncoder 或 cv2 编码 JPEG
    Cam->>DTPS: 发布图像 / 相机内参（配置 SHM 失败时回退 HTTP）
    Whl->>DTPS: dtps_init() 创建 in/pwm、in/estop 与 out/pwm_* 队列
    DTPS-->>Whl: 投递 DifferentialPWM（cb_wheels_pwm）与 Boolean（cb_estop）
    Whl->>Drv: 未触发 estop 时调用 set_wheels_speed(left, right)
    Whl->>DTPS: 发布 out/pwm_filtered 与 out/pwm_executed 状态
```

- **最短复现路径：** 通过 `README.md` 中的 `duckietown/duckiebot-interface:ente-arm64v8` 容器启动脚本（如 `launchers/sensor-camera.sh`、`launchers/actuator-wheels.sh`）或 `packages/duckiebot_interface/include/duckiebot_interface/launch/main.py` 按硬件配置拉起各驱动节点，经 DTPS 队列完成感知发布与底盘 PWM 闭环。
- **本次核查源码文件（`duckietown/dt-duckiebot-interface` `ente`）：** `README.md`、`launchers/default.sh`、`launchers/actuator-wheels.sh`、`launchers/sensor-camera.sh`、`packages/duckiebot_interface/include/duckiebot_interface/launch/main.py`、`packages/wheels_driver/include/wheels_driver_node/main.py`、`packages/camera_driver/include/camera_driver/camera_driver_abs.py`、`packages/camera_driver/include/camera_driver_node/raspberry_pi_64.py`。

## 工程实践

- 先从 dt-duckiebot-interface 与 dt-core 入手，在 gym-duckietown 里调通再上真车。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [turtlebot3](./turtlebot3.md)
- [paper-openbot-smartphone-mobile-robot](./paper-openbot-smartphone-mobile-robot.md)
- [sim2real](../concepts/sim2real.md)
- [paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat](./paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)
- [Sim-to-Real Transfer Workflow · Sim2Real 迁移工作流](../methods/sim2real-transfer-workflow.md)：对应从 gym-duckietown 仿真环境调试策略后迁移至 Raspberry Pi 真车与真实道路系统的验证工作流。
- [多智能体强化学习（MARL） | Multi-Agent Reinforcement Learning](../methods/marl.md)：对应基于 dt-core 模块化软件栈与多车道路场景开展多智能体强化学习教学与实验的研究方向。

## 推荐继续阅读

- [Duckietown 组织](https://github.com/duckietown)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
