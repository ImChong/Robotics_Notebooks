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
  - ../../sources/repos/uniarml1.md
summary: "宇树科技 开源项目 UniArmL1（遥操作与数据采集）：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据…"
institutions:
  - unitree
---

# UniArmL1

## 一句话定义

**UniArmL1** 是 [宇树科技](https://github.com/unitreerobotics) 公开的 **遥操作与数据采集** 开源项目：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

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
| 机构 | 宇树科技 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/unitreerobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `UniArmL1` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python teleop/teleop.py -i vr --record --task-dir ./data/pick_place --task-goal "pick up the cup"`（`teleop/teleop.py` 的 `main`；`-i` 可选 `vr` / `keyboard` / `leader`）

```mermaid
sequenceDiagram
    participant U as 用户 shell
    participant T as teleop/teleop.py main
    participant A as arm/uniarm_l1 UniArmL1
    participant BUS as arm/uniarm_l1_bus UnitreeMotorsBus
    participant IN as input VRInput / KeyboardInput / LeaderArmInput
    participant R as teleop_runner TeleopRunner
    participant KIN as arm/kinematics RobotKinematics
    participant REC as recorder EpisodeRecorder
    participant EW as utils/episode_writer EpisodeWriter
    participant CAM as image_server/utils cameras
    Note over U,T: 主从模式（-i leader）时另建一个 UniArmL1 作主臂
    U->>T: python teleop/teleop.py -i vr --record
    T->>A: UniArmL1(follower_config)
    A->>BUS: UnitreeMotorsBus（no_real_robot 或连接失败则 bus 为空，进入仿真）
    A->>CAM: make_cameras_from_configs
    T->>IN: 构造输入源
    T->>REC: EpisodeRecorder 或 NullRecorder
    T->>R: TeleopRunner(robot, input_source, recorder, ...)
    T->>R: setup() 然后 run()
    R->>A: start_control_loop（bus 存在时启动后台线程）
    loop 每个控制周期 control_dt
        R->>IN: get_q_sim（主从）或 get_delta（VR / 键盘）
        R->>A: step(active, dxyz, drot, trigger)（增量模式）
        A->>KIN: inverse_kinematics 与 forward_kinematics
        R->>A: map_sim_to_output_rad_all 写 tgt_pos_rad
        R->>CAM: cam.async_read 取各相机帧
        R->>REC: start_episode 或 add_item（按 record_hz 录制）
        REC->>EW: create_episode 与 add_item
    end
    Note over A,BUS: 后台线程 _control_loop_worker 周期调用 _position_control_step，写电机目标并 close_control
    R->>REC: stop_episode 与 close（_cleanup）
    REC->>EW: save_episode
```

图依据：UniArmL1@cae412d，README.md（运行命令与参数表）、teleop/teleop.py（读过 main 的装配段，argparse 仅 grep）、teleop/robot_control/teleop_runner.py（setup、run、_compute_q_sim 与 _cleanup）、teleop/robot_control/arm/uniarm_l1.py（__init__ 的 bus 与 cameras 装配、start_control_loop、_control_loop_worker、_position_control_step、step）、input/input_vr.py 与 input/input_leader.py（仅读取 get_delta、get_q_sim 相关片段）；uniarm_l1_bus.py、recorder.py、utils/episode_writer.py、image_server/utils.py 仅 grep 类与函数定义。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [UniArmL1 源码归档](../../sources/repos/uniarml1.md)（<https://github.com/unitreerobotics/UniArmL1>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：unitreerobotics/UniArmL1@cae412d，读过 README.md、teleop/teleop.py、teleop/robot_control/teleop_runner.py、teleop/robot_control/arm/uniarm_l1.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [宇树科技 官方组织](https://github.com/unitreerobotics)
