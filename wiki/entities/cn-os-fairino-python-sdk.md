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
  - ../../sources/repos/fairino-python-sdk.md
summary: "法奥意威 开源项目 fairino-python-sdk（SDK/驱动）：提供法奥意威协作机器人Python控制SDK，支持状态读取、运动命令和应用程序集成。…"
institutions:
  - fair-innovation
---

# fairino-python-sdk

## 一句话定义

**fairino-python-sdk** 是 [法奥意威](https://github.com/FAIR-INNOVATION) 公开的 **SDK/驱动** 开源项目：提供法奥意威协作机器人Python控制SDK，支持状态读取、运动命令和应用程序集成。

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
| 机构 | 法奥意威 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/FAIR-INNOVATION |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `fairino-python-sdk` 仓库并核对 README 许可与依赖。
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

入口：`python linux/example/TestBasicCommand.py`（示例脚本 `from fairino import Robot; robot = Robot.RPC('192.168.58.2')`，库入口为 `linux/fairino/Robot.py` 的 `class RPC`）

```mermaid
sequenceDiagram
    participant E as example/TestBasicCommand.py
    participant R as fairino/Robot.py RPC.__init__
    participant N as FRCNDEClient（TCP 20005）
    participant X as xmlrpc.client.ServerProxy（http://ip:20003）
    participant U as FrUdpClient（UDP 20007）
    participant C as 法奥机器人控制器

    E->>R: Robot.RPC(ip)
    R->>N: connect(ip, 20005)
    N->>C: 发送 CNDE 输出配置、启动（_send_cnde_output_config / _set_cnde_start）
    N->>N: 启动 CNDERecvThread，持续写入 robot_state_pkg
    R->>X: GetControllerIP() 连通性测试
    X->>C: XML-RPC 请求
    R->>X: GetTLSEnableState()，与 SDK 的 tls_enable 比对
    R->>U: FrUdpClient(ip, 20007).start_recv_thread()
    R->>R: RPC.is_connect = True
    E->>R: robot.RobotEnable(1)
    R->>X: self.robot.RobotEnable(state)（socket.error 时循环重试）
    X->>C: 使能
    E->>R: robot.MoveJ(joint_pos, tool, user, ...)
    R->>R: GetSafetyCode()（读 robot_state_pkg 的 safety_stop0/1 状态，非 0 直接返回）
    R->>X: GetForwardKin(joint_pos)（desc_pos 全零时补算位姿）
    R->>X: self.robot.MoveJ(joint_pos, desc_pos, tool, user, vel, acc, ovl, ...)
    X->>C: 执行关节运动
    C-->>N: CNDE 状态流（关节位置、安全状态等）
```

图依据：FAIR-INNOVATION/fairino-python-sdk@8920448，读过 `README.md`、`linux/fairino/README.txt`、`linux/fairino/Robot.py`（`RPC.__init__`、`FRCNDEClient.connect`、`xmlrpc_timeout`、`RobotEnable`、`MoveJ`、`GetSafetyCode`、`GetActualJointPosDegree`）、`linux/example/TestBasicCommand.py`（头部）。UDP 帧（`FrUdpClient`）的收发细节与 mTLS/DTLS 分支未读；`windows/` 目录未读。机器人端协议实现为闭源控制器，此处只描述 SDK 一侧。

## 参考来源

- [fairino-python-sdk 源码归档](../../sources/repos/fairino-python-sdk.md)（<https://github.com/FAIR-INNOVATION/fairino-python-sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：FAIR-INNOVATION/fairino-python-sdk@8920448，读过 README.md、linux/fairino/README.txt、linux/fairino/Robot.py（RPC.__init__、FRCNDEClient.connect、RobotEnable、MoveJ、GetSafetyCode 等）、linux/example/TestBasicCommand.py（头部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [法奥意威 官方组织](https://github.com/FAIR-INNOVATION)
