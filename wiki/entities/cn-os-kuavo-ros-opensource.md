---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/kuavo-ros-opensource.md
summary: "乐聚机器人 开源项目 kuavo-ros-opensource（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - leju
---

# kuavo-ros-opensource

## 一句话定义

**kuavo-ros-opensource** 是 [乐聚机器人](https://github.com/LejuRobotics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 乐聚机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/LejuRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `kuavo-ros-opensource` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch`（先 `source devel/setup.bash`；轮臂机型用 `load_kuavo_mujoco_sim_wheel.launch`）。该 launch 拉起 `humanoid_sqp_mpc`、`nodelet_manager`、`HumanoidControllerNodelet`，并通过 `mujoco_cpp` 的 `nodelet_with_arm.launch` 加载 `MujocoNodelet`。以下仅画双足（`nodelet_robot_type=2`）分支。

```mermaid
sequenceDiagram
    participant L as roslaunch load_kuavo_mujoco_sim.launch
    participant MPC as humanoid_sqp_mpc (HumanoidSqpMpcNode.cpp)
    participant NM as nodelet_manager
    participant CN as HumanoidControllerNodelet (controllerNodelet.cpp)
    participant HC as humanoidController (humanoidController.cpp)
    participant WBC as humanoid_wbc (wbc_)
    participant MJ as MujocoNodelet (mujoco_node.cc)

    L->>MPC: 启动 humanoid_sqp_mpc (use_external_mpc=true)
    L->>NM: 启动 nodelet_manager
    L->>CN: nodelet load HumanoidControllerNodelet
    L->>MJ: nodelet_with_arm.launch: load MujocoNodelet
    CN->>HC: onInit: 启动 controlLoop 线程 (SCHED_FIFO)
    HC->>HC: init: setupMpc 创建 mrtRosInterface_, 加载 WBC 任务
    MJ-->>HC: /sensors_data_raw (sensorsDataCallback)
    loop 控制循环 (controlLoop 调用 update)
        HC->>HC: updateStateEstimation (IMU / 运动学 / EKF)
        HC->>MPC: setCurrentObservation (MRT 观测)
        MPC->>MPC: SqpMpc 求解最优控制问题
        MPC-->>HC: MPC 策略 (spinMRT / updatePolicy / evaluatePolicy)
        HC->>WBC: wbc_->update(optimizedState, optimizedInput, measuredRbdState, plannedMode)
        WBC-->>HC: 关节加速度 / 力矩 x
        HC->>HC: publishControlCommands: 填充 jointCmd
        HC->>MJ: /joint_cmd (kuavo_msgs/jointCmd)
        MJ->>MJ: jointCmdCallback 缓存指令
        MJ->>MJ: 仿真循环 mj_step 推进一步
        MJ-->>HC: /sensors_data_raw (下一拍传感器)
    end
```

图依据：LejuRobotics/kuavo-ros-opensource@34bd149，`readme.md`、`src/humanoid-control/humanoid_controllers/launch/load_kuavo_mujoco_sim.launch`、`src/humanoid-control/humanoid_controllers/src/controllerNodelet.cpp`、`src/humanoid-control/humanoid_controllers/src/humanoidController.cpp`（update / init / publishControlCommands 等段落）、`src/humanoid-control/humanoid_interface_ros/src/HumanoidSqpMpcNode.cpp`、`src/mujoco/launch/nodelet_with_arm.launch`、`src/mujoco/src/mujoco_node.cc`（节选）、`src/mujoco/src/mujoco_nodelet.cc`。

## 参考来源

- [kuavo-ros-opensource 源码归档](../../sources/repos/kuavo-ros-opensource.md)（<https://github.com/LejuRobotics/kuavo-ros-opensource>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LejuRobotics/kuavo-ros-opensource@34bd149，读过 readme.md、src/humanoid-control/humanoid_controllers/launch/load_kuavo_mujoco_sim.launch、src/humanoid-control/humanoid_controllers/src/controllerNodelet.cpp、src/humanoid-control/humanoid_controllers/src/humanoidController.cpp（部分段落）、src/humanoid-control/humanoid_interface_ros/src/HumanoidSqpMpcNode.cpp（部分）、src/mujoco/launch/nodelet_with_arm.launch、src/mujoco/src/mujoco_node.cc（节选）、src/mujoco/src/mujoco_nodelet.cc（检索）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [乐聚机器人 官方组织](https://github.com/LejuRobotics)
