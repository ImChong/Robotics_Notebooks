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
  - ../../sources/repos/magiclab_deploy.md
summary: "魔法原子 开源项目 magiclab_deploy（运动RL/技能训练）：提供魔法原子运动策略的运行与部署入口，将策略输出接到机器人控制接口。…"
institutions:
  - magiclab
---

# magiclab_deploy

## 一句话定义

**magiclab_deploy** 是 [魔法原子](https://github.com/MagiclabRobotics) 公开的 **运动RL/技能训练** 开源项目：提供魔法原子运动策略的运行与部署入口，将策略输出接到机器人控制接口。

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
| 机构 | 魔法原子 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/MagiclabRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `magiclab_deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python deploy_real_z1.py --config z1_12dof.yaml`（aarch64 可用 `bash run_nx.sh`；需先设置 `magicbot-z1` SDK 的 `PYTHONPATH` / `LD_LIBRARY_PATH`）。

```mermaid
sequenceDiagram
    participant M as deploy_real_z1.py main()
    participant Cfg as config_z1.Config (configs/z1_12dof.yaml)
    participant Ctl as MagicBotController
    participant SDK as magicbot_z1_python (MagicRobot / 低层控制器)
    participant ODE as ort.InferenceSession odom
    participant POL as ort.InferenceSession policy
    M->>Cfg: Config("configs/z1_12dof.yaml")
    M->>Ctl: MagicBotController(config)
    Ctl->>POL: 加载 config.policy_path（CPUExecutionProvider）
    Ctl->>ODE: 加载 config.odom_path
    Ctl->>SDK: initialize_robot：MagicRobot.initialize("192.168.54.111") / connect()
    Ctl->>SDK: set_motion_control_level(LowLevel) / get_low_level_motion_controller()
    Ctl->>SDK: subscribe_body_imu / leg / arm / waist / head state（回调写入缓存）
    M->>Ctl: start_joystick_thread()（手柄或 SSH 键盘读指令）
    M->>Ctl: zero_torque_state()
    Ctl->>SDK: publish_*_command（kp=kd=toq=0），循环直到 cmd_btn[0] 启动指令
    M->>Ctl: move_to_default_pos()
    Ctl->>SDK: 插值发送默认姿态 PD 指令
    M->>Ctl: default_pos_state()，等待进入 ACTIVE_CONTROL
    loop 2 ms 周期（500 Hz）while state == ACTIVE_CONTROL
        M->>Ctl: run_active_control()
        Ctl->>Ctl: get_joint_positions / velocities / get_imu_data
        alt 每 control_decimation 步（注释标 50 Hz）
            Ctl->>ODE: 滚动历史 odom 观测 → 估计基座线速度
            Ctl->>POL: 滚动历史 policy 观测 → 动作 self.action
        end
        Ctl->>SDK: send_position_commands → publish_leg/arm/waist/head_command
    end
    M->>Ctl: 退出键后 move_to_default_pos()；finally 中 enter_damping_state() / shutdown()
    Ctl->>SDK: disconnect() / shutdown()
```

图依据：magiclab_deploy@5f58d6a，读过 README.md、deploy_real_z1.py（initialize_robot、send_zero_torque_commands、run_active_control、main 等）。config_z1.py、configs/z1_12dof.yaml、gamepad_reader_btp.py、run_nx.sh 未读内部。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [magiclab_deploy 源码归档](../../sources/repos/magiclab_deploy.md)（<https://github.com/MagiclabRobotics/magiclab_deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：MagiclabRobotics/magiclab_deploy@5f58d6a，读过 README.md、deploy_real_z1.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [魔法原子 官方组织](https://github.com/MagiclabRobotics)
