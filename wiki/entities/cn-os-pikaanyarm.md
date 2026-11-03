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
  - ../../sources/repos/pikaanyarm.md
summary: "松灵机器人 开源项目 PikaAnyArm（遥操作与数据采集）：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据…"
institutions:
  - agilex
---

# PikaAnyArm

## 一句话定义

**PikaAnyArm** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **遥操作与数据采集** 开源项目：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 松灵机器人 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `PikaAnyArm` 仓库并核对 README 许可与依赖。
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

入口：`roslaunch pika_remote_piper teleop_rand_single_piper.launch`（Piper 单臂；该 launch 先 include `piper/launch/start_single_piper.launch`，再起 `piper_FK.py`、`piper_IK.py`、`teleop_piper_publish.py` 三个节点；`/pika_pose` 由仓库外的 pika_ros 提供）

```mermaid
sequenceDiagram
    participant P as Pika Sense（仓库外，发布 /pika_pose）
    participant T as teleop_piper_publish.py
    participant FK as piper_FK.py
    participant IK as piper_IK.py（Arm_IK，pinocchio）
    participant D as piper_ctrl_single_node.py
    participant R as Piper 机械臂（piper_sdk C_PiperInterface，CAN）

    D->>R: C_PiperInterface(can_port).ConnectPort()，auto_enable 时 EnableArm
    loop 约 200 Hz（Pubilsh）
        D-->>FK: 发布 /joint_states_single
    end
    loop 约 200 Hz（calc）
        FK->>FK: arm_fk.get_pose(关节 1~6)
        FK-->>T: /piper_FK/urdf_end_pose_orient
    end
    Note over T: teleop_trigger 服务（std_srvs/Trigger）触发后，status_changing 在两个位姿回调中各锁存一次基准位姿，随后 status=True
    P-->>T: /pika_pose（localization_pose_callback）
    T->>T: 目标位姿 = 机械臂基准末端位姿 x inv(Pika 基准位姿) x 当前 Pika 位姿
    T-->>IK: /piper_IK/ctrl_end_pose
    T-->>T: 同时发布 /teleop_status（TeleopStatus）
    IK->>IK: arm_end_pose_callback：pin.SE3 目标，arm_ik.ik_fun
    alt 有解且位置偏差不超过 0.3
        IK-->>IK: ArmControlStatus over_limit=False，更新 sol_q
        loop 约 50 Hz（publishing）
            IK-->>D: /joint_states（线性插值 ctrl，单步超过 30 度时按 1 度分步 200 Hz）
        end
    else 无解或偏差过大
        IK-->>IK: 发布 /arm_control_status over_limit=True
    end
    Note over D,IK: start_single_piper.launch 把 joint_ctrl_single 重映射为 /joint_states_gripper，而 IK 发布的是 /joint_states，二者如何接通本仓库内未看到
    D->>R: joint_callback：MotionCtrl_2 然后 JointCtrl（夹爪 GripperCtrl）
```

图依据：agilexrobotics/PikaAnyArm@d2b8b84，读过 `README.md`、`piper/pika_remote_piper/launch/teleop_rand_single_piper.launch`、`piper/pika_remote_piper/scripts/{teleop_piper_publish.py,piper_FK.py,piper_IK.py}`、`piper/piper_ros/piper/launch/start_single_piper.launch`、`piper/piper_ros/piper/scripts/piper_ctrl_single_node.py`（仅看了发布/订阅/SDK 调用行）。仓库另含 `ur/`、`xarm/` 两套同构遥操作包，本图未覆盖。

## 参考来源

- [PikaAnyArm 源码归档](../../sources/repos/pikaanyarm.md)（<https://github.com/agilexrobotics/PikaAnyArm>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：agilexrobotics/PikaAnyArm@d2b8b84，读过 README.md、piper/pika_remote_piper/launch/teleop_rand_single_piper.launch、piper/pika_remote_piper/scripts/{teleop_piper_publish.py,piper_FK.py,piper_IK.py}、piper/piper_ros/piper/launch/start_single_piper.launch、piper/piper_ros/piper/scripts/piper_ctrl_single_node.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
