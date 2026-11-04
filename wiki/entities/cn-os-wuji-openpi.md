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
  - ../../sources/repos/wuji-openpi.md
summary: "舞肌科技 开源项目 wuji-openpi（VLA/操作模型）：扩展OpenPI以支持双臂与双Wuji Hand配置，连接ROS2 MCAP示范、LeRobot数据转换、pi0或pi0.5监督微调、策略服务和真机推理。…"
institutions:
  - wuji-robotics
---

# wuji-openpi

## 一句话定义

**wuji-openpi** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **VLA/操作模型** 开源项目：扩展OpenPI以支持双臂与双Wuji Hand配置，连接ROS2 MCAP示范、LeRobot数据转换、pi0或pi0.5监督微调、策略服务和真机推理。

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
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wuji-openpi` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：服务端 `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_wuji_multi_54d --policy.dir=<checkpoint>`；机器人端 `python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml`（ROS2 环境内运行）。

```mermaid
sequenceDiagram
    autonumber
    participant S as scripts/serve_policy.py 服务端入口
    participant WS as websocket_policy_server.py WebsocketPolicyServer
    participant P as policies/policy.py Policy
    participant C as wuji_policy.py WujiInputs/WujiOutputs
    participant M as examples/wuji/deploy/main.py 机器人端入口
    participant W as websocket_client_policy.py WebsocketClientPolicy
    participant B as action_chunk_broker.py ActionChunkBroker
    participant R as runtime/runtime.py Runtime
    participant A as runtime/agents/policy_agent.py PolicyAgent
    participant E as examples/wuji/deploy/ros_env.py OpenPIRosEnvironment

    S->>P: create_trained_policy 加载训练配置与 checkpoint
    S->>WS: WebsocketPolicyServer(policy, port).serve_forever()
    M->>M: parse_args 合并 deploy.yaml 与命令行参数
    M->>W: WebsocketClientPolicy(host, port) 连接服务端
    W->>WS: 建立 websocket 连接
    WS-->>W: 发送 policy metadata
    M->>E: OpenPIRosEnvironment(node, config) 等待 ROS2 关节与图像数据
    M->>B: 按 broker_mode 构造 ActionChunkBroker 或 RTG 系 broker
    M->>R: Runtime(environment, PolicyAgent(broker), max_hz, ...)
    M->>R: runtime.run()
    loop 每个 episode 的每个控制步
        R->>E: get_observation()
        E-->>R: observation/state, prompt, 头部与腕部图像
        R->>A: get_action(observation)
        A->>B: infer(observation)
        alt 当前 action chunk 已用尽
            B->>W: infer(obs)
            W->>WS: msgpack 打包 obs 并发送
            WS->>P: policy.infer(obs)
            P->>C: 输入变换 WujiInputs
            P->>P: model.sample_actions 采样动作块
            P->>C: 输出变换 WujiOutputs
            P-->>WS: actions 与 policy_timing
            WS-->>W: 返回 action chunk 与 server_timing
            W-->>B: 解包 action chunk
        end
        B-->>A: 取出当前步动作
        A-->>R: action
        R->>E: apply_action(action)
        E->>E: publish_action 发布 ROS2 关节命令
    end
    R->>E: reset() 回到初始状态
```

图依据：wuji-technology/wuji-openpi@be6b325，实际读过 `README.md`、`scripts/serve_policy.py`、`src/openpi/serving/websocket_policy_server.py`、`src/openpi/policies/policy.py`、`src/openpi/policies/wuji_policy.py`、`examples/wuji/deploy/main.py`、`examples/wuji/deploy/ros_env.py`、`packages/openpi-client/src/openpi_client/{websocket_client_policy,action_chunk_broker}.py`、`packages/openpi-client/src/openpi_client/runtime/{runtime,agents/policy_agent}.py`。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wuji-openpi 源码归档](../../sources/repos/wuji-openpi.md)（<https://github.com/wuji-technology/wuji-openpi>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：wuji-technology/wuji-openpi@be6b325，读过 README.md、scripts/serve_policy.py、src/openpi/serving/websocket_policy_server.py、src/openpi/policies/policy.py、src/openpi/policies/wuji_policy.py、examples/wuji/deploy/main.py、examples/wuji/deploy/ros_env.py、packages/openpi-client 中的 websocket_client_policy / action_chunk_broker / runtime 模块；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
