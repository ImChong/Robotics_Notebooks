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
  - ../../sources/repos/wuji-hand-teleop.md
summary: "舞肌科技 开源项目 wuji-hand-teleop（遥操作与数据采集）：通过ROS2接入多种人体输入设备并实时控制Wuji Hand和机器人手臂，为示范采集和操作闭环提供统一入口。…"
institutions:
  - wuji-robotics
---

# wuji-hand-teleop

## 一句话定义

**wuji-hand-teleop** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **遥操作与数据采集** 开源项目：通过ROS2接入多种人体输入设备并实时控制Wuji Hand和机器人手臂，为示范采集和操作闭环提供统一入口。

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
| 机构 | 舞肌科技 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wuji-hand-teleop` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch wuji_teleop_bringup wuji_teleop_hand.launch.py`（`src/wuji_teleop_bringup/launch/wuji_teleop_hand.launch.py`；仓库 README 另给 Monitor GUI 一键入口 `ros2 run wuji_teleop_monitor monitor`，本图未画）

```mermaid
sequenceDiagram
    participant U as 用户 shell
    participant L as wuji_teleop_hand.launch.py
    participant DRV as wujihandros2 wujihand_driver_node
    participant CN as controller wujihand_node WujiHandControllerNode
    participant SDK as wuji_sdk SdkManager 进程内 UDP
    participant WO as wujihand_output WujiHandController
    participant RT as wuji_retargeting Retargeter
    participant HI as wujihand_output WujiHand
    U->>L: ros2 launch wuji_teleop_hand.launch.py
    L->>DRV: 为左右手各启动一个 wujihand_driver_node（按 hand_name 分命名空间）
    L->>CN: 启动两个 wujihand_controller 进程（--side left 与 --side right）
    CN->>WO: 创建 WujiHandController（input_source 为 wuji_glove）
    WO->>RT: Retargeter.from_yaml 读取 retarget_wuji_glove_{side}.yaml
    WO->>HI: 创建 WujiHand，其发布者话题为 /{hand_name}/joint_commands
    CN->>SDK: _setup_wuji_glove 与 _connect_glove 连接手套（lazy import）
    CN->>CN: create_timer 按 control_rate（默认 120 Hz）调用 _teleop_loop
    loop 每个控制周期
        CN->>SDK: _sdk_sub.recv 取最新 HandSkeleton，丢弃积压旧帧
        CN->>CN: _extract_wuji_glove_keypoints 转为 (21, 3) 关键点
        CN->>WO: set_keypoints(kp)
        WO->>RT: retarget(keypoints) 得到关节角
        WO->>HI: set_joint_positions(angles)
        HI->>DRV: 发布 /{hand_name}/joint_commands（JointState）
    end
    Note over CN,SDK: 接收超时则 _connect_glove 重连（看门狗，_RECV_TIMEOUT_SEC）
```

图依据：wuji-hand-teleop@6478013，README.md（启动方式与节点表）、src/wuji_teleop_bringup/launch/wuji_teleop_hand.launch.py（完整读过）、src/controller/controller/wujihand_node.py（__init__、_teleop_loop 与手套分支、main 段）、src/controller/setup.py（仅 grep console_scripts）、src/output_devices/wujihand_output/wujihand_output/wujihand_controller.py（set_keypoints 与 retarget）、_internal/hand_interface.py（仅 grep 与 set_joint_positions 片段）。launch 文件经 launch_utils.read_input_source 读取 input_source，该函数未读。wuji_teleop_monitor 未读。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wuji-hand-teleop 源码归档](../../sources/repos/wuji-hand-teleop.md)（<https://github.com/wuji-technology/wuji-hand-teleop>）

- 源码核对（2026-11-02）：wuji-technology/wuji-hand-teleop@6478013，读过 README.md、src/wuji_teleop_bringup/launch/wuji_teleop_hand.launch.py、src/controller/controller/wujihand_node.py、src/output_devices/wujihand_output/wujihand_output/wujihand_controller.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
