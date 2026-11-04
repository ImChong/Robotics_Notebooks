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
  - ../../sources/repos/engineai_ros2_workspace.md
summary: "众擎机器人 开源项目 engineai_ros2_workspace（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - engineai
---

# engineai_ros2_workspace

## 一句话定义

**engineai_ros2_workspace** 是 [众擎机器人](https://github.com/engineai-robotics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 众擎机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/engineai-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `engineai_ros2_workspace` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（先 `./scripts/build_nodes.sh sim` 并 `source install/setup.bash`）：`ros2 launch mujoco_simulator mujoco_simulator.launch.py`

```mermaid
sequenceDiagram
    autonumber
    participant LF as mujoco_simulator.launch.py
    participant MN as mujoco/src/main.cc main
    participant SM as sim_manager.cc SimManager
    participant CL as config_loader.cc ConfigLoader
    participant RI as ros_interface.cc RosInterface
    participant PT as SimManager PhysicsLoop 物理线程
    participant MJ as MuJoCo mj_step 与 mjcb_control
    participant UI as simulate Simulate RenderLoop 主线程
    participant CLI as interface_example joint_bridge_example.py 客户端
    LF->>MN: 设置 PRODUCT=pm_v2、MUJOCO_ASSETS_PATH, 启动 mujoco_simulator 节点
    MN->>SM: rclcpp::init, GetInstance(), Initialize()
    SM->>CL: 读取 assets/config/pm_v2.yaml (urdf, xml, 话题名, num_total_joints)
    SM->>RI: make_unique RosInterface(node, config), Initialize()
    RI->>RI: publisher: /hardware/joint_state, /hardware/imu_info, /motion/motion_state
    RI->>RI: subscriber: /hardware/joint_command (best_effort, KeepLast 1)
    SM->>MJ: mjcb_control = TorqueControllerWrapper
    SM->>UI: new Simulate(GlfwAdapter, cam, opt, pert)
    MN->>SM: Run()
    SM->>PT: std::thread 启动 PhysicsThread(model_file)
    SM->>UI: sim_->RenderLoop() 在主线程运行
    PT->>MJ: mj_loadXML(assets/resource/pm_v2.xml), mj_makeData, mj_forward
    PT->>RI: SetModelAndData(m, d)
    loop PhysicsLoop, 直到 exitrequest
        PT->>RI: rclcpp::spin_some(节点)
        CLI->>RI: /hardware/joint_command (JointCommand)
        RI->>RI: JointCommandCallback 加锁写入 joint_command_
        PT->>MJ: mj_step(m, d)
        MJ->>SM: 每步调用 mjcb_control, 即 TorqueController(m, d)
        SM->>RI: GetCommandedSafe() 取得命令副本
        SM->>MJ: d.ctrl[i] = 前馈力矩 + 刚度 x 位置误差 + 阻尼 x 速度误差
        PT->>RI: UpdateSimState(m, d) 取关节与 IMU 数据
        RI-->>CLI: /hardware/joint_state 与 /hardware/imu_info
    end
    RI-->>CLI: /motion/motion_state (每 1 秒 wall timer)
```

图依据：engineai-robotics/engineai_ros2_workspace@c41043f，README.md、src/simulation/mujoco/launch/mujoco_simulator.launch.py、src/simulation/mujoco/src/main.cc、sim_manager.cc、src/simulation/mujoco/src/ros_interface.cc（节选）、config_loader.cc（节选）、src/simulation/mujoco/include/ros_interface.h（检索）、src/simulation/mujoco/assets/config/pm_v2.yaml、src/interface_example/scripts/joint_bridge_example.py（节选）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [engineai_ros2_workspace 源码归档](../../sources/repos/engineai_ros2_workspace.md)（<https://github.com/engineai-robotics/engineai_ros2_workspace>）

- 源码核对（2026-11-02）：engineai-robotics/engineai_ros2_workspace@c41043f，读过 README.md、src/simulation/mujoco/{launch/mujoco_simulator.launch.py,src/main.cc,src/sim_manager.cc,src/ros_interface.cc,src/config_loader.cc,assets/config/pm_v2.yaml}、src/interface_example/scripts/joint_bridge_example.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [众擎机器人 官方组织](https://github.com/engineai-robotics)
