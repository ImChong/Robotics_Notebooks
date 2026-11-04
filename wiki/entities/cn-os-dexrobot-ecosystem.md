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
  - ../../sources/repos/dexrobot_ecosystem.md
summary: "灵巧智能 开源项目 dexrobot_ecosystem（移动操作）：整合灵巧手底层控制、运动学、URDF、Isaac Sim、MuJoCo和ROS兼容层，连接真实硬件、仿真与操作算法开发。…"
institutions:
  - psibot
---

# dexrobot_ecosystem

## 一句话定义

**dexrobot_ecosystem** 是 [灵巧智能](https://github.com/DexRobot) 公开的 **移动操作** 开源项目：整合灵巧手底层控制、运动学、URDF、Isaac Sim、MuJoCo和ROS兼容层，连接真实硬件、仿真与操作算法开发。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第三层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 灵巧智能 |
| 类别 | 移动操作 |
| 官方组织 | https://github.com/DexRobot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `dexrobot_ecosystem` 仓库并核对 README 许可与依赖。
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

入口：`python examples/ros_node/dexhand_ros.py`（ROS 节点，读取 `config/config.yaml` 中的 `DexHand.ROS_Node` 配置）；硬件自检入口为 `python tools/hardware_test/test_dexhand.py --hands right`。

```mermaid
sequenceDiagram
    autonumber
    participant ROS as ROS 话题 /joint_commands
    participant N as DexHandNode dexhand_ros.py
    participant JM as JointMapping
    participant H as LeftDexHand RightDexHand dexhand_interface.py
    participant Z as ZCANWrapper zcan_wrapper.py
    participant CAN as libusbcanfd USBCANFD 适配器

    N->>H: hand_class(zcan) 构造左右手对象
    N->>H: init() 打开 ZCAN 设备并配置通道
    H->>Z: open 与 configure_channel
    N->>N: create_subscription 订阅 /joint_commands
    N->>N: create_service 注册 reset_hands
    N->>N: create_timer 以 rate 周期调用 send_commands
    ROS->>N: JointState 消息
    N->>N: command_callback 解析 name 与 position
    N->>JM: map_command(joint_values) 映射到硬件关节
    JM-->>N: 硬件关节指令
    N->>N: 低通滤波 alpha 更新 last_commands

    loop 每个定时周期 send_commands
        N->>H: move_joints(th_dip, th_mcp, ff_mcp, ...)
        H->>H: _scale_angle 角度转电机位置并生成使能掩码
        loop 每块板 board_idx
            H->>H: _send_motion_command 编码 CANFD 报文
            H->>Z: send_fd_message(channel, command_id, data)
            Z->>CAN: 通过 libusbcanfd 发送 CANFD 帧
        end
    end
    Note over H,Z: 反馈链路 get_feedback 到 receive_fd_messages 未在 ROS 节点中调用
```

图依据：DexRobot/dexrobot_ecosystem@3d0700c 的子模块 `pyzlg_dexhand@316575a`（gitee，`https://gitee.com/DexRobot/pyzlg_dexhand.git`），读过 `README.md`、`docs/quickstart.rst`（前 80 行）、`examples/ros_node/dexhand_ros.py`（导入、`__init__`、`command_callback`、`send_commands`、`main` 段）、`pyzlg_dexhand/dexhand_interface.py`（`init`、`move_joints`、`_refresh_board_states` 段及方法列表）、`pyzlg_dexhand/zcan_wrapper.py`（类与方法列表）；`ros_compat` 与其它子模块（dexrobot_urdf / kinematics / mujoco / isaac）未读。时序图依据以上文件，未运行代码。

## 参考来源

- [dexrobot_ecosystem 源码归档](../../sources/repos/dexrobot_ecosystem.md)（<https://github.com/DexRobot/dexrobot_ecosystem>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：DexRobot/dexrobot_ecosystem@3d0700c 与子模块 pyzlg_dexhand@316575a，读过主仓库 `README.md`、`setup.py`、`.gitmodules`、`.github/workflows/build.yml`，子模块 `README.md`、`docs/quickstart.rst`（部分）、`examples/ros_node/dexhand_ros.py`、`pyzlg_dexhand/dexhand_interface.py`（部分）、`pyzlg_dexhand/zcan_wrapper.py`（grep）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [灵巧智能 官方组织](https://github.com/DexRobot)
