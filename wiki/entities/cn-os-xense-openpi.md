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
  - ../../sources/repos/xense-openpi.md
summary: "千觉机器人 开源项目 xense-openpi（工程与工具）：触觉数据接入 OpenPI（pi0）训练与推理链路…"
institutions:
  - qianjue
---

# xense-openpi

## 一句话定义

**xense-openpi** 是 [千觉机器人](https://github.com/XenseRobotics-AI) 公开的 **工程与工具** 开源项目：触觉数据接入 OpenPI（pi0）训练与推理链路

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
| 机构 | 千觉机器人 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/XenseRobotics-AI、https://github.com/XenseRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `xense-openpi` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：服务端 `python scripts/serve_policy.py --env DROID`（`--env DROID` 时加载 `pi05_droid` 默认检查点；`--env` 缺省为 ALOHA_SIM；端口默认 8000）；客户端 `python examples/simple_client/main.py --env DROID`（随机观测压测，见 README）。

```mermaid
sequenceDiagram
    autonumber
    participant SP as scripts/serve_policy.py main
    participant PC as policy_config.py create_trained_policy
    participant WS as websocket_policy_server.py WebsocketPolicyServer
    participant PO as policies/policy.py Policy.infer
    participant M as 模型 sample_actions
    participant CL as examples/simple_client/main.py
    participant WC as xense_client WebsocketClientPolicy

    SP->>SP: tyro.cli(Args) 解析 env / port / policy
    SP->>PC: create_policy → create_default_policy → create_trained_policy(config, dir)
    PC->>PC: download.maybe_download；加载 params 与 norm_stats
    PC-->>SP: Policy(model, input/output transforms)
    SP->>WS: WebsocketPolicyServer(policy, host 0.0.0.0, port, metadata)
    SP->>WS: serve_forever → asyncio.run(run)
    WS->>WS: websockets.serve(_handler, process_request=_health_check)
    CL->>WC: WebsocketClientPolicy(host, port)
    WC->>WS: 建立连接，_wait_for_server 收 metadata
    WS-->>WC: msgpack 打包的 metadata
    loop 每个控制步
        CL->>WC: infer(obs_fn())
        WC->>WS: ws.send(msgpack 打包的 obs)
        WS->>WS: unpackb，弹出 __rtc_kwargs__
        WS->>PO: infer(obs, **rtc_kwargs)
        PO->>PO: 输入变换：默认 prompt、数据变换、Normalize
        PO->>M: sample_actions(observation, sample_kwargs)
        M-->>PO: action chunk
        PO->>PO: 输出变换：Unnormalize、数据变换输出
        PO-->>WS: actions dict
        WS->>WS: 写入 server_timing，msgpack 打包
        WS-->>WC: 发送 action 帧
        WC-->>CL: unpackb 后的 action
    end
    alt 推理异常
        WS-->>WC: 发送 traceback 字符串并以 INTERNAL_ERROR 关闭
        WC-->>CL: raise RuntimeError
    end
```

图依据：xense-openpi@6b9a30f，实际读过 README.md、scripts/serve_policy.py、src/openpi/serving/websocket_policy_server.py、src/openpi/policies/policy_config.py（create_trained_policy 段）、src/openpi/policies/policy.py（infer 段）、packages/xense-client/src/xense_client/websocket_client_policy.py（infer 与连接段）、examples/simple_client/main.py（main 循环）、examples/simple_client/README.md；未读 Policy 内 sample_actions 的具体扩散/流匹配实现与 models/ 下各模型文件。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [xense-openpi 源码归档](../../sources/repos/xense-openpi.md)（<https://github.com/XenseRobotics-AI/xense-openpi>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：XenseRobotics-AI/xense-openpi@6b9a30f，读过 README.md、scripts/serve_policy.py、src/openpi/serving/websocket_policy_server.py、src/openpi/policies/policy_config.py、src/openpi/policies/policy.py、packages/xense-client/src/xense_client/websocket_client_policy.py、examples/simple_client/main.py、examples/simple_client/README.md；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [千觉机器人 官方组织](https://github.com/XenseRobotics-AI)
