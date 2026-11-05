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
  - ../../sources/repos/agibot_x1_infer.md
summary: "智元机器人 开源项目 agibot_x1_infer（部署运行时）：策略部署运行时：加载训练策略、下发关节命令…"
institutions:
  - agibot
---

# agibot_x1_infer

## 一句话定义

**agibot_x1_infer** 是 [智元机器人](https://github.com/AgibotTech) 公开的 **部署运行时** 开源项目：策略部署运行时：加载训练策略、下发关节命令

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
| 机构 | 智元机器人 |
| 类别 | 部署运行时 |
| 官方组织 | https://github.com/AgibotTech、https://github.com/AimRT、https://github.com/OpenDriveLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `agibot_x1_infer` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./aimrt_main --cfg_file_path=./cfg/x1_cfg.yaml`（`src/install/linux/bin/run.sh`；仿真用 `run_sim.sh` + `x1_cfg_sim.yaml`）。`aimrt_main` 是 AimRT 运行时，按配置从 `libpkg1.so`（`src/pkg/pkg1/pkg_main.cc`）加载 `JoyStickModule`、`ControlModule`、`DcuDriverModule` 三个模块，模块间通过 AimRT channel（`/cmd_vel_limiter`、`/imu/data`、`/joint_states`、`/joint_cmd` 等话题）通信。

```mermaid
sequenceDiagram
    participant Main as aimrt_main (x1_cfg.yaml)
    participant Joy as JoyStickModule (joy_stick_module.cc)
    participant Ctl as ControlModule (control_module.cc)
    participant RL as RLController (rl_controller.cc)
    participant Dcu as DcuDriverModule (dcu_driver_module.cc)
    participant Xyber as XyberController (EtherCAT)

    Main->>Joy: Initialize / Start，executor joy_stick_pub_thread 跑 MainLoop
    Main->>Ctl: Initialize 读 rl_x1.yaml，建 state_machine_ 与各 controller，订阅话题
    Main->>Dcu: Initialize 读 dcu_x1.yaml，InitTransmission，订阅 /joint_cmd
    Main->>Ctl: Start，executor rl_control_pub_thread 跑 MainLoop
    Main->>Dcu: Start，起 PublishLoop 线程
    Dcu->>Xyber: Start(ifname, cycle_time_ns)，EnableActuator
    loop 每个手柄周期
        Joy->>Ctl: Publish Twist 到 /cmd_vel_limiter，按键 Float32 到 /walk_mode 等 trigger_topic
    end
    Ctl->>Ctl: 触发回调 state_machine_.OnEvent(trigger_topic)，对当前 controller 调 RestartController
    Ctl->>RL: 回调 SetCmdData(Twist)
    loop PublishLoop
        Dcu->>Xyber: GetPosition / GetVelocity / GetEffort
        Dcu->>Dcu: transmission_.TransformActuatorToJoint
        Dcu->>Ctl: Publish /joint_states 与 /imu/data
    end
    Ctl->>RL: 回调 SetJointStateData / SetImuData（先减 joint_offset）
    loop MainLoop，control_frequecy=1000 Hz
        Ctl->>RL: Update()
        RL->>RL: UpdateStateEstimation
        Note over RL: 每 decimation 拍：ComputeObservation（sin/cos 相位、速度指令、关节、上次动作、IMU）
        RL->>RL: ComputeActions，ONNX session_ptr_->Run(rl_walk_leg.onnx)，裁剪 actions_clip
        Ctl->>RL: GetJointCmdData()，pos_des = action*action_scale+init_state，低通后发出
        Ctl->>Dcu: Publish /joint_cmd (JointCommand)
        Dcu->>Dcu: JointCmdCallback，transmission_.TransformJointToActuator
        Dcu->>Xyber: SetMitCmd(name, pos, vel, effort, kp, kd)
    end
```

图依据：AgibotTech/agibot_x1_infer@9e0b818，读过 README.md、src/install/linux/bin/run.sh、cfg/x1_cfg.yaml、src/pkg/pkg1/pkg_main.cc、src/module/control_module/{src/control_module.cc,src/rl_controller.cc,cfg/rl_x1.yaml}、src/module/dcu_driver_module/src/dcu_driver_module.cc、src/module/joy_stick_module/src/joy_stick_module.cc（`/imu/data` 在 DcuDriverModule 内发布；XyberController 内部实现与 `sim_module` 未读）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [agibot_x1_infer 源码归档](../../sources/repos/agibot_x1_infer.md)（<https://github.com/AgibotTech/agibot_x1_infer>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：AgibotTech/agibot_x1_infer@9e0b818，读过 README.md、src/install/linux/bin/run.sh、cfg/x1_cfg.yaml、src/pkg/pkg1/pkg_main.cc、control_module/{control_module.cc,rl_controller.cc,rl_x1.yaml}、dcu_driver_module.cc、joy_stick_module.cc；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智元机器人 官方组织](https://github.com/AgibotTech)
