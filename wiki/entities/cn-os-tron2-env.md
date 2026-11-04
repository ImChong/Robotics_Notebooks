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
  - ../../sources/repos/tron2_env.md
summary: "逐际动力 开源项目 tron2_env（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - limx
---

# tron2_env

## 一句话定义

**tron2_env** 是 [逐际动力](https://github.com/limxdynamics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 逐际动力 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/limxdynamics、https://github.com/FluxVLA |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `tron2_env` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（无需机器人）：`python examples/mock_quickstart.py`（真机路径为 `create_motion_controller(config, backend="websocket")`，transport 换成 WebsocketTransport）

```mermaid
sequenceDiagram
    autonumber
    participant EX as examples/mock_quickstart.py main
    participant RT as RecordingTransport（内存替身，仓库内示例）
    participant MC as MotionController（motion/controller.py）
    participant AS as ServoJStrategy（插值与发布原语）
    participant TH as MotionController-publish 线程

    EX->>RT: RecordingTransport()
    EX->>MC: MotionController(transport, publish_rate=100, eta_default=0.02)
    EX->>MC: start()
    MC->>AS: read_measured_target，经 transport 读取测量状态，reset 插值器
    MC->>TH: 启动 _publish_loop 线程
    EX->>MC: command_joints(target 16 维)
    MC->>MC: command_arm 拆为左右臂 7 维与头部 2 维
    MC->>AS: set_destination(target, eta)
    loop 每 1 / publish_rate 秒
        TH->>AS: publish_current()，取插值当前值
        AS->>RT: send_joint_cmd(q)，记录到 sent_frames
    end
    EX->>MC: disconnect()
    EX->>RT: sent_frames()，断言最后一帧与 target 一致
    Note over EX,RT: 真机路径：Tron2Env.reset / step 调用同一 MotionController，send_joint_cmd 经 WebsocketTransport 发送 request_servoj 到 ws://robot_ip:5000
```

图依据：limxdynamics/tron2_env@7553ca6，examples/mock_quickstart.py、src/tron2_env/motion/controller.py、src/tron2_env/motion/arm_strategy.py（publish_current 段）、src/tron2_env/env.py（reset 与 step 段）、src/tron2_env/transport/websocket.py（send_joint_cmd 段）、README.md

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [tron2_env 源码归档](../../sources/repos/tron2_env.md)（<https://github.com/limxdynamics/tron2_env>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：limxdynamics/tron2_env@7553ca6，读过 README.md、examples/mock_quickstart.py、src/tron2_env/motion/controller.py（start、command_joints、_publish_loop 与工厂函数段）、src/tron2_env/motion/arm_strategy.py（publish_current 段）、src/tron2_env/env.py（reset 与 step 段）、src/tron2_env/transport/websocket.py（send_joint_cmd 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [逐际动力 官方组织](https://github.com/limxdynamics)
