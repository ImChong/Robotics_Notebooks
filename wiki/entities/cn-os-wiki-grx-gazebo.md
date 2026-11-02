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
  - ../../sources/repos/wiki-grx-gazebo.md
summary: "傅利叶智能 开源项目 Wiki-GRx-Gazebo（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - fourier
---

# Wiki-GRx-Gazebo

## 一句话定义

**Wiki-GRx-Gazebo** 是 [傅利叶智能](https://github.com/FFTAI) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

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
| 机构 | 傅利叶智能 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/FFTAI |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Wiki-GRx-Gazebo` 仓库并核对 README 许可与依赖。
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

入口：`roslaunch rl_sim gazebo_gr1t1.launch`（或 `gazebo_gr1t2.launch`；ROS Noetic + catkin，需 libtorch）

```mermaid
sequenceDiagram
    participant L as rl_sim/launch/gazebo_gr1t1.launch
    participant G as Gazebo（empty_world + stairs.world）
    participant CM as controller_manager spawner
    participant JC as robot_joint_controller（RobotJointController 插件）
    participant RS as rl_sim 节点（rl_sim.cpp: RL_Sim）
    participant K as 键盘线程 KeyboardInterface
    participant SC as RL::StateController（rl_sdk.cpp）
    participant P as TorchScript 策略 model_4000_jit.pt

    L->>G: include empty_world.launch（paused=true）
    L->>G: spawn_model 载入 GR1T1_lower_limb.urdf（z=1.0）
    L->>CM: 加载 robot_control.yaml，spawn joint_state_controller 与 10 个 *_controller
    CM->>JC: 实例化每个关节控制器，订阅 <joint>/command
    L->>RS: 启动 rl_sim 节点
    RS->>RS: ReadYaml(config.yaml[robot_name])；torch::jit::load(models/<robot>/model_4000_jit.pt)
    RS->>RS: 订阅 /cmd_vel、/gazebo/model_states、joint_states；启动 loop_keyboard / loop_control(dt) / loop_rl(dt*decimation)
    K->>RS: Enter 切换仿真；0 起立；p 进入 RL；w/a/s/d/j/l 改速度指令
    RS->>G: /gazebo/unpause_physics（Enter 后）
    loop loop_control 每 dt (config.yaml: 0.001 s)
        RS->>RS: GetState（model_states 的 IMU 姿态/角速度 + joint_states 的 q、dq、effort）
        RS->>SC: StateController(state, command)
        SC-->>RS: GETUP 阶段线性插值到 default_dof_pos；RL_RUNNING 阶段用 output_dof_pos 与 rl_kp/rl_kd
        RS->>JC: 发布 robot_msgs/MotorCommand（q、dq、kp、kd、tau）
        JC->>G: update(): tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff，joint.setCommand
    end
    loop loop_rl 每 dt*decimation (0.001*20 s)
        RS->>RS: 填充 obs（角速度、四元数、commands、dof_pos、dof_vel）
        RS->>P: Forward(): ComputeObservation 后 model.forward
        P-->>RS: 动作，clamp 后 ComputePosition 得 output_dof_pos
    end
```

图依据：FFTAI/Wiki-GRx-Gazebo@3617042，读过 `README.md`、`src/rl_sim/launch/gazebo_gr1t1.launch`、`src/rl_sim/config.yaml`（gr1t1 段）、`src/rl_sim/src/rl_sim.cpp`、`src/rl_sim/library/rl_sdk/rl_sdk.cpp`（StateController、KeyboardInterface、ReadYaml 部分）、`src/robot_joint_controller/src/robot_joint_controller.cpp`（update 与订阅部分）、`src/robots/gr1t1_description/config/robot_control.yaml`（头部）。`ComputePosition`/`ComputeTorques` 内部、`rl_sdk.hpp`、`loop.hpp` 与 gr1t2 的差异未读；STATE_RL_RUNNING 之后的 GETDOWN 分支未读完。

## 参考来源

- [Wiki-GRx-Gazebo 源码归档](../../sources/repos/wiki-grx-gazebo.md)（<https://github.com/FFTAI/Wiki-GRx-Gazebo>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：FFTAI/Wiki-GRx-Gazebo@3617042，读过 README.md、src/rl_sim/launch/gazebo_gr1t1.launch、src/rl_sim/config.yaml、src/rl_sim/src/rl_sim.cpp、src/rl_sim/library/rl_sdk/rl_sdk.cpp（部分）、src/robot_joint_controller/src/robot_joint_controller.cpp（部分）、src/robots/gr1t1_description/config/robot_control.yaml（头部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傅利叶智能 官方组织](https://github.com/FFTAI)
