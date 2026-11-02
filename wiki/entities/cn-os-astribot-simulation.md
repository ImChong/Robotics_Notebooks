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
  - ../../sources/repos/astribot_simulation.md
summary: "星尘智能 开源项目 astribot_simulation（仿真环境）：提供星尘智能机器人相关的仿真开发环境，用于加载模型并验证机器人系统接口。…"
institutions:
  - astribot
---

# astribot_simulation

## 一句话定义

**astribot_simulation** 是 [星尘智能](https://github.com/Astribot-Dev) 公开的 **仿真环境** 开源项目：提供星尘智能机器人相关的仿真开发环境，用于加载模型并验证机器人系统接口。

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
| 机构 | 星尘智能 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/Astribot-Dev |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `astribot_simulation` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`source env.sh && python3 astribot_simulation.py`（默认读 `config/astribot_s1/simulation_mujoco_param.yaml`）

```mermaid
sequenceDiagram
    participant Main as astribot_simulation.py main()
    participant Fac as AstribotEnvsFactory (astribot_envs_factory.py)
    participant Gym as gymnasium.make
    participant Env as AstribotMujocoEnv (astribot_mujoco_env.py)
    participant Base as AstribotBaseEnv (astribot_base_env.py)
    participant Ros as MultiRobotRosInterface / RobotRosInterface (robot_ros_interface.py)
    participant MJ as mujoco (mj_step)

    Main->>Fac: load_yaml_file(yaml)（model_path 拼上 ASTRIBOT_SIMU_ROOT）
    Main->>Fac: create_simulation_env(data)
    Fac->>Gym: make("astribot_envs/AstribotMujocoEnv-v0", param=data)
    Gym->>Env: __init__(param)
    Env->>Base: super().__init__：读 YAML 参数、setup_ros_node（rospy/rclpy init）
    Base->>Ros: setup_joint_interface：建 joint_space_states 发布与 joint_space_command 订阅
    Env->>MJ: MjModel.from_xml_path / MjData，建 MujocoRenderer
    Fac->>Fac: 起线程运行 simu_env_loop（Isaac Lab 则主线程）
    Fac->>Env: reset()
    loop while running()（ROS rate 50Hz）
        Fac->>Env: step(action_space.sample())
        Env->>Base: update_reset_flag / update_joint_states
        Base->>Ros: publish_joint_states（仿真关节状态发布）
        Ros-->>Base: get_joint_position/velocity/torque_command（来自 joint_command_callback）
        Env->>Env: 按 controller_mode 1/2/3 设置 data.ctrl（含重力补偿、速度补偿）
        loop frame_skip=3
            Env->>MJ: mj_step(model, data)
        end
        Env->>Env: render()（human 窗口或异步相机渲染）
    end
    Fac->>Env: close()
```

图依据：Astribot-Dev/astribot_simulation@1403f96，读过 `README.md`、`astribot_simulation.py`、`src/astribot_envs/astribot_envs_factory.py`、`astribot_mujoco_env.py`、`astribot_base_env.py`、`src/simu_utils/robot_ros_interface.py`、`env.sh`。仅画 MuJoCo 后端默认路径；Genesis / ManiSkill / Isaac Lab 后端未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [astribot_simulation 源码归档](../../sources/repos/astribot_simulation.md)（<https://github.com/Astribot-Dev/astribot_simulation>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Astribot-Dev/astribot_simulation@1403f96，读过 README.md、astribot_simulation.py、src/astribot_envs/astribot_envs_factory.py、astribot_mujoco_env.py、astribot_base_env.py、src/simu_utils/robot_ros_interface.py、env.sh；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [星尘智能 官方组织](https://github.com/Astribot-Dev)
