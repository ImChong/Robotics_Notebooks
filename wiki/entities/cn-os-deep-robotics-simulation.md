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
  - ../../sources/repos/deep-robotics-simulation.md
summary: "云深处科技 开源项目 deep-robotics-simulation（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - deeprobotics
---

# deep-robotics-simulation

## 一句话定义

**deep-robotics-simulation** 是 [云深处科技](https://github.com/DeepRoboticsLab) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

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
| 机构 | 云深处科技 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/DeepRoboticsLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `deep-robotics-simulation` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python3 run_sim.py dr02_pro`（`dr02_std` 同构，调用 `dr02_std/mujoco_simulation.py`）

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as run_sim.py main()
    participant P as dr02_pro/mujoco_simulation.py run()
    participant C as sim_config.py load_config()
    participant N as DR2ProSimulation (rclpy Node)
    participant S as simulator.py MuJoCoSimulator
    participant M as MuJoCo mj_step / viewer
    participant X as 外部控制器 (ROS 2 Topic)

    U->>R: python3 run_sim.py dr02_pro
    R->>P: run(dr02_pro/config.yaml)
    P->>P: rclpy.init()
    P->>C: 读取 config.yaml（timestep_s、publish_frequency_hz、command_timeout_s、control.mode、viewer）
    C-->>P: SimulationConfig（含 scene.xml 路径）
    P->>N: DR2ProSimulation(config)
    N->>S: MuJoCoSimulator(scene_path, config)
    S->>M: MjModel.from_xml_path / MjData / launch_passive
    N->>N: _validate_model()（29 个执行器与关节映射、upper 约束、IMU 传感器）
    N->>S: reset_to_keyframe("default")
    N->>X: 创建发布者 /JOINTS_DATA /IMU_DATA_HEAD /IMU_DATA_BASE 与订阅 /JOINTS_CMD，create_timer(publish_frequency_hz)
    P->>N: simulation.run()
    loop rclpy.ok() 且未请求停止
        N->>N: rclpy.spin_once(timeout_sec=0)
        X-->>N: JointsCmd → _handle_joint_command（写入 goal_joint_pos/vel、kp、kd、tau_ff）
        N->>N: _publish_state()（定时器回调）
        N-->>X: 发布 Joints 与 Imu（来自 qpos/qvel 与 base_quat/base_acc/base_omega 传感器）
        alt 尚未收到有效指令
            N->>S: sync_viewer()，sleep(timestep_s)
        else 指令超时 command_timeout_s
            N->>N: _apply_joint_command() 置 ctrl 为 0（upper 模式暂停物理）
        else 正常
            N->>N: _apply_joint_command()：tau = kp*(q_goal-q) + kd*(dq_goal-dq) + tau_ff
            N->>S: step()（按 timestep 限速）
            S->>M: mujoco.mj_step
            N->>S: sync_viewer()
        end
    end
    U->>P: SIGINT/SIGTERM → request_stop()
    N->>S: close()
```

图依据：DeepRoboticsLab/deep-robotics-simulation@52902fc，读过 README.md、run_sim.py、sim_config.py、simulator.py、dr02_pro/mujoco_simulation.py、dr02_pro/config.yaml。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [deep-robotics-simulation 源码归档](../../sources/repos/deep-robotics-simulation.md)（<https://github.com/DeepRoboticsLab/deep-robotics-simulation>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：DeepRoboticsLab/deep-robotics-simulation@52902fc，读过 README.md、run_sim.py、sim_config.py、simulator.py、dr02_pro/mujoco_simulation.py、dr02_pro/config.yaml；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [云深处科技 官方组织](https://github.com/DeepRoboticsLab)
