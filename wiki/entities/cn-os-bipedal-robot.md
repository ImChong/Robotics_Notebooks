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
  - ../../sources/repos/bipedal-robot.md
summary: "达妙科技 开源项目 bipedal-robot（工程与工具）：公开双足机器人本体与控制相关资料，为执行器、结构和运动控制实验提供基础。…"
institutions:
  - dmbots
---

# bipedal-robot

## 一句话定义

**bipedal-robot** 是 [达妙科技](https://github.com/dmBots) 公开的 **工程与工具** 开源项目：公开双足机器人本体与控制相关资料，为执行器、结构和运动控制实验提供基础。

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
| 机构 | 达妙科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/dmBots |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `bipedal-robot` 仓库并核对 README 许可与依赖。
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

入口：`roslaunch legged_controllers one_start_real.launch`（真机；仿真为 `one_start_gazebo.launch`），其中 `load_controller_real.launch` 负责加载 `controllers/legged_controller`。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant L as one_start_real.launch
    participant HW as legged_dm_hw main + DmHW
    participant TD as timedelay_launch.sh
    participant LC as load_controller_real.launch
    participant CM as controller_manager spawner
    participant LCtl as LeggedController
    participant EST as FromTopicStateEstimate
    participant MPC as SqpMpc + MPC_MRT_Interface
    participant HWL as LeggedHWLoop

    U->>L: roslaunch one_start_real.launch
    L->>HW: include legged_dm_hw.launch 启动 legged_dm_hw
    HW->>HW: DmHW::init 初始化 LeggedHW 与 dmbot_serial 电机接口
    L->>TD: 延时 2 秒后执行 load_controller_real.launch
    TD->>LC: 启动
    LC->>CM: spawner 加载 controllers/legged_controller
    CM->>LCtl: init(robot_hw, nh)
    LCtl->>LCtl: setupLeggedInterface 读取 urdf 与 task.info
    LCtl->>MPC: setupMpc 创建 SqpMpc 与 RosReferenceManager
    LCtl->>EST: setupStateEstimate 加载估计器
    CM->>LCtl: starting(time) 设置初始目标与 ModeSubscribe
    LCtl->>MPC: 启动 mpcThread_ 循环 advanceMpc

    loop 控制循环 LeggedHWLoop::update
        HWL->>HW: read() 读取电机 pos vel tau 与 IMU
        HWL->>CM: update(time, period)
        CM->>LCtl: update(time, period)
        LCtl->>EST: updateJointStates / updateImu / update
        LCtl->>MPC: setCurrentObservation updatePolicy evaluatePolicy
        MPC-->>LCtl: optimizedState 与 optimizedInput
        LCtl->>LCtl: 力映射 tor = Jfoot 转置乘接触力 并做关节限位检查
        LCtl->>HW: hybridJointHandles setCommand(pos, vel, kp, kd, ff)
        HWL->>HW: write() 下发电机指令 send_motor_data
    end
```

图依据：dmBots/bipedal-robot@957cbcc，`src/代码/上位机代码/MPC方案代码/src/` 下的 `legged_controllers/launch/one_start_real.launch`、`one_start_gazebo.launch`、`load_controller_real.launch`、`legged_controllers/src/LeggedController.cpp`（init / starting / update / updateStateEstimation / setupMpc / mpcThread_ 相关段）、`legged_hw/src/LeggedHWLoop.cpp`、`legged_examples/legged_dm_hw/src/legged_dm_hw.cpp`、`legged_examples/legged_dm_hw/src/DmHW.cpp`（read / write 段）、`legged_estimation/src/FromTopicEstimate.cpp`（update 段）。

## 参考来源

- [bipedal-robot 源码归档](../../sources/repos/bipedal-robot.md)（<https://github.com/dmBots/bipedal-robot>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：dmBots/bipedal-robot@957cbcc，读过 `README.md`（顶层，仅 Gitee 模板）、`src/代码/上位机代码/MPC方案代码/README.md`、上述 launch 文件、`LeggedController.cpp`、`LeggedHWLoop.cpp`、`legged_dm_hw.cpp`、`DmHW.cpp`、`FromTopicEstimate.cpp`、`legged_hw/CMakeLists.txt`、`TargetTrajectoriesPublisher.cpp`（grep 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [达妙科技 官方组织](https://github.com/dmBots)
