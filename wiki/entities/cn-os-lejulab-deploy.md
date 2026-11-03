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
  - ../../sources/repos/lejulab-deploy.md
summary: "乐聚机器人 开源项目 LejuLab-Deploy（运动RL/技能训练）：策略部署运行时：加载训练策略、下发关节命令…"
institutions:
  - leju
---

# LejuLab-Deploy

## 一句话定义

**LejuLab-Deploy** 是 [乐聚机器人](https://github.com/LejuRobotics) 公开的 **运动RL/技能训练** 开源项目：策略部署运行时：加载训练策略、下发关节命令

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
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/LejuRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `LejuLab-Deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`export ROBOT_VERSION=46 && roslaunch leju_launch load_mujoco_sim.launch`（实物为 `load_real.launch`）；launch 文件拉起 `leju-mujoco-sim`、`leju-joystick`、`run_rl_controller`（`leju-rl-controller/src/main.cpp`）、`lejusdk_recorder`。仓库 `src/` 内只有 RL 控制器、IK、VR、音频等源码，`leju-mujoco-sim`、`leju-joystick` 以预编译二进制放在 `installed/lib/`，其内部实现未开源，下图只画 `run_rl_controller` 一侧。

```mermaid
sequenceDiagram
    participant L as leju_launch/load_mujoco_sim.launch
    participant M as run_rl_controller（src/main.cpp）
    participant RD as RobotData
    participant CM as ControllerManager
    participant CL as runtime/ControlLoop
    participant RL as GenericRLController
    participant IT as 推理线程（inferenceThreadLoop）
    participant SDK as lejusdk GlobalRobot（DDS）
    participant SIM as leju-mujoco-sim（预编译）

    L->>M: 启动 -c controller_manager.yaml -u urdf -t teleop_bindings.yaml
    M->>SDK: GlobalRobot::init_env(RobotVersion::from_env())
    M->>RD: robot_data.initialize()（订阅状态 / IMU）
    M->>CM: initialize(config_file, urdf_path)，按 yaml 加载控制器（默认 default_controller）
    M->>M: TeleopInputSource.initialize()，手柄与 Quest 触发写入 TriggerBuffer
    M->>CM: Start()，waitForDataReady()
    M->>RL: moveToDefaultPos(state, 3.0)
    M->>CL: control_loop.run()
    loop 每个控制周期（tick，周期由当前控制器 waitNextCycle 决定）
        CL->>RD: getSynchronizedData(state, imu)
        SIM-->>SDK: 关节状态 / IMU
        SDK-->>RD: 订阅回调写入
        CL->>CL: drainAll 触发器，Lifecycle.update（start / back）
        CL->>CL: mergeAllCmdVel 等按输入源优先级合并命令
        CL->>CM: update(state, imu, merged_cmd)
        CM->>RL: setVelocityCommand，update
        alt 每 decimation 步
            RL->>RL: computeObservation
            RL->>IT: submitObservationForInference（条件变量唤醒）
            IT->>IT: inferActions → InferenceModel::forward（ONNX / OpenVINO）
            IT-->>RL: updateActionCache
        end
        RL->>RL: updateRobotCmd：q_target = direction × (base_pos + action × scale)
        RL-->>CM: RobotCmd
        CM-->>CL: RobotCmd
        CL->>SDK: GlobalRobot::publishRobotCmd(cmd)
        SDK-->>SIM: 关节指令
    end
```

图依据：LejuLab-Deploy@212353e，读过 `README.md`、`src/leju_launch/launch/load_mujoco_sim.launch`、`load_real.launch`（节点列表）、`src/leju-controllers/leju-rl-controller/src/main.cpp`、`src/runtime/control_loop.cpp`（run、tick、publishRobotCmd）、`src/controllers/controller_manager.cpp`（update，约第 525 至 680 行）、`src/controllers/generic_rl_controller.cpp`（updateImpl、updateRobotCmd、submitObservationForInference、inferActions、inferenceThreadLoop）、`config/46/controller_manager.yaml`（头部）。SDK 内部（`lejusdk-lowlevel`）、模型加载细节、`mimic` 等其他控制器分支未读；`SIM`/`SDK` 之间的数据流向依据 README 架构图与 `installed/` 二进制清单推断，未读其源码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [LejuLab-Deploy 源码归档](../../sources/repos/lejulab-deploy.md)（<https://github.com/LejuRobotics/LejuLab-Deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LejuRobotics/LejuLab-Deploy@212353e，读过 README.md、src/leju_launch/launch/{load_mujoco_sim,load_real}.launch、src/leju-controllers/leju-rl-controller/src/{main.cpp,runtime/control_loop.cpp,controllers/controller_manager.cpp,controllers/generic_rl_controller.cpp}、config/46/controller_manager.yaml；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [乐聚机器人 官方组织](https://github.com/LejuRobotics)
