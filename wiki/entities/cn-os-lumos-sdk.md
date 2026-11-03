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
  - ../../sources/repos/lumos_sdk.md
summary: "鹿明机器人 开源项目 lumos_sdk（SDK/驱动）：提供鹿明机器人C++集成接口，使状态、设备与控制功能接入上层系统。…"
institutions:
  - lumos
---

# lumos_sdk

## 一句话定义

**lumos_sdk** 是 [鹿明机器人](https://github.com/LumosRobot) 公开的 **SDK/驱动** 开源项目：提供鹿明机器人C++集成接口，使状态、设备与控制功能接入上层系统。

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
| 机构 | 鹿明机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/LumosRobot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `lumos_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cmake -S . -B build && cmake --build build` 后运行 `./build/nix_joint_cmd --component WAIST --hold-seconds 0.5`（Python 等价入口：`python3 python/nix_debug_state.py enter` 与 `python3 python/nix_joint_cmd.py single ...`）。机器人侧 `lumos_controller` 不在本仓库内，时序图中该参与者的行为仅依据 README 描述。

```mermaid
sequenceDiagram
    participant Ex as example/nix_joint_cmd.cpp main
    participant Mgr as SdkRobotManager (src/sdk_robot_manager.cpp)
    participant Bus as LCM 组播 (lcm_robot_cmd / lcm_joint_cmd 等)
    participant Ctl as 机器人侧 lumos_controller（README 描述，非本仓库）
    Ex->>Mgr: Init()（等待 lcm_.good()，启动 HandleLcmRecv 线程）
    Ex->>Mgr: SetRobotStatusCb / SetImuDataCb / SetJointDataCb
    Mgr->>Bus: subscribe lcm_robot_status / lcm_imu_data / lcm_joint_data
    Bus-->>Ex: IMU 回调累计 g_imu_count（最多等 3 秒，为 0 则退出）
    Ex->>Mgr: SendRobotCmd(RESET)（ensure_reset_briefly）
    Mgr->>Bus: publish lcm_robot_cmd / robot_cmd_lcmt
    Bus->>Ctl: 状态切换请求
    Ctl-->>Ex: lcm_robot_status（wait_state 确认）
    Ex->>Mgr: SendRobotCmd(STAND)
    Mgr->>Bus: publish lcm_robot_cmd
    Ctl-->>Ex: lcm_robot_status = STAND(2)
    Note over Ex: 等待 kStandSettleSec=11 秒让站立插值完成
    Ex->>Mgr: SendRobotCmd(DEBUG)
    Mgr->>Bus: publish lcm_robot_cmd
    Ctl-->>Ex: lcm_robot_status = DEBUG(10)（wait_state 超时则不发关节命令）
    loop 100 Hz（kControlHz）：从当前 pos_high 插值到目标位置
        Ex->>Mgr: SendJointCmds(vector<SdkJointCmd>)
        Mgr->>Bus: publish lcm_joint_cmd / joint_cmds_lcmt
        Bus->>Ctl: 关节目标（仅 DEBUG 状态生效）
        Ctl-->>Ex: lcm_joint_data / joint_datasets_lcmt 更新 g_current_pos
    end
```

图依据：lumos_sdk@43c2a80，读过 README.md、CMakeLists.txt、src/sdk_robot_manager.cpp、example/nix_joint_cmd.cpp（main 流程）、python/nix_debug_state.py、python/nix_robot_state.py、python/nix_joint_cmd.py（函数签名级）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [lumos_sdk 源码归档](../../sources/repos/lumos_sdk.md)（<https://github.com/LumosRobot/lumos_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LumosRobot/lumos_sdk@43c2a80，读过 README.md、CMakeLists.txt、src/sdk_robot_manager.cpp、example/nix_joint_cmd.cpp、python/nix_debug_state.py、python/nix_robot_state.py、python/nix_joint_cmd.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [鹿明机器人 官方组织](https://github.com/LumosRobot)
