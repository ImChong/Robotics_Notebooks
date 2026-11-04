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
  - ../../sources/repos/seersdk4j.md
summary: "仙工智能 开源项目 SeerSdk4j（SDK/驱动）：为仙工智能移动机器人控制器提供Java TCP客户端与请求示例，便于业务系统读取状态和调用控制接口。…"
institutions:
  - seer
---

# SeerSdk4j

## 一句话定义

**SeerSdk4j** 是 [仙工智能](https://github.com/seer-robotics) 公开的 **SDK/驱动** 开源项目：为仙工智能移动机器人控制器提供Java TCP客户端与请求示例，便于业务系统读取状态和调用控制接口。

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
| 机构 | 仙工智能 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/seer-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `SeerSdk4j` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`com.seer.sdk.demo.RbkClientDemo.main`（`src/test/java/com/seer/sdk/demo/RbkClientDemo.java`，以 Java 主方法运行；README 未给出构建或运行命令，示例 IP 为 `192.168.8.114`）

```mermaid
sequenceDiagram
    participant Demo as RbkClientDemo.main
    participant Cli as RbkClient
    participant Port as RbkPortClient（状态口 19204）
    participant Enc as RbkEncoder
    participant Net as NettyClient / Netty 通道
    participant Dec as RbkDecoder + RbkInboundHandler
    participant Bot as 仙工机器人控制器（TCP）
    Demo->>Cli: new RbkClient(host)，建立 5 个 RbkPortClient（19207 / 19210 / 19204 / 19205 / 19206）
    Demo->>Cli: request(1007, {"simple": true}, 10000)
    Cli->>Cli: 按 apiNo 区间分派：1000–1999 → stateClient
    Cli->>Port: stateClient.request → doRequest（ReentrantLock）
    Port->>Port: getNettyClient：首次创建 Bootstrap.connect，pipeline 为 RbkDecoder + RbkInboundHandler
    Port->>Enc: buildReqBytes(apiNo, body, flowNo)，头部含起始符 / 版本 / flowNo / 长度 / apiNo
    Enc-->>Port: CompositeByteBuf
    Port->>Net: write → writeAndFlush，await 5 秒
    Net->>Bot: 请求帧
    Bot-->>Dec: 响应帧
    Dec->>Dec: decode：读取头部（≥15 字节），按 bodySize 切出 RbkFrame
    Dec->>Port: channelRead0 → onMessage：responseMap.put(flowNo, body)，condition.signalAll
    Port->>Port: condition.await(timeout) 直到 responseMap 含本 flowNo
    Port-->>Cli: RbkResult（kind = Ok，resStr）
    Cli-->>Demo: RbkResult
    Demo->>Demo: kind Ok 时解析 ret_code 与 battery_level 或 err_msg；否则取 errMsg
    Demo->>Cli: dispose()，释放各 RbkPortClient
    Note over Port: 非 Ok（Timeout / ConnectFail / WriteError 等）时 reset() 销毁该端口连接
```

图依据：SeerSdk4j@c19a0ab，`src/test/java/com/seer/sdk/demo/RbkClientDemo.java`、`src/main/java/com/seer/sdk/rbk/RbkClient.java`（构造、request 分派、dispose）、`RbkPortClient.java`（request / doRequest / createNettyClient / onMessage）、`RbkEncoder.java`（buildReqBytes）、`RbkDecoder.java`（decode）、`RbkInboundHandler.java`（channelRead0）、`NettyClient.java`（write）、`README.md`；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [SeerSdk4j 源码归档](../../sources/repos/seersdk4j.md)（<https://github.com/seer-robotics/SeerSdk4j>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：seer-robotics/SeerSdk4j@c19a0ab，读过 README.md、src/test/java/com/seer/sdk/demo/RbkClientDemo.java、src/main/java/com/seer/sdk/rbk/ 下 RbkClient、RbkPortClient、RbkEncoder、RbkDecoder、RbkInboundHandler、NettyClient 的相关段落；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [仙工智能 官方组织](https://github.com/seer-robotics)
