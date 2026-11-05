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
  - ../../sources/repos/multimotortcpapi.md
summary: "钛虎机器人 开源项目 multiMotorTCPAPI（SDK/驱动）：多电机 TCP 控制 API…"
institutions:
  - ti5robot
---

# multiMotorTCPAPI

## 一句话定义

**multiMotorTCPAPI** 是 [钛虎机器人](https://github.com/ti5robot) 公开的 **SDK/驱动** 开源项目：多电机 TCP 控制 API

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
| 机构 | 钛虎机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/ti5robot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `multiMotorTCPAPI` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`sudo ./multi_motor`（README「2.3 编译」：由 `src/main.cpp` 编译生成，编译命令见 `g++ CMD .txt`）

仓库只提供头文件与预编译 `.so`，`socket_server` 与 `login` 等函数的实现未随源码提供，下图只画 `main.cpp` 中可见的调用顺序，不推断库内部调用。

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant M as main.cpp 的 main()
    participant B as Ti5BASIC.h 声明的函数
    participant SV as socket_server（socket_service.h 声明，实现未提供）
    participant CL as 外部 TCP 客户端
    U->>M: sudo ./multi_motor
    M->>M: signal(SIGINT, signalHandler)
    M->>M: canidList = new uint8_t[Num_Len]，依次赋值 1 到 Num_Len
    M->>B: login() 登录 CAN 设备
    M->>SV: socket_server("192.168.130.64", 12345) 以服务端方式启动
    Note over SV: 头文件注释称接收数据去掉帧头帧尾后交给 socket_parse_data 转换为角度值，此处仅据注释，未画内部调用
    CL->>SV: 客户端连接并发送数据（仓库只有 socket_client.h 头文件，无客户端源码）
    SV-->>M: 返回 0 成功或 1 失败
    M->>B: deallocate_variable()
    M->>B: logout() 登出 CAN 设备
    Note over M: Ctrl+C 触发 signalHandler: brake(Num_Len) → logout() → exit(signum)
```

图依据：ti5robot/multiMotorTCPAPI@ce2295f，读过 README.md、src/main.cpp、include/socket_service.h、include/socket_client.h、include/Ti5BASIC.h（声明部分）、g++ CMD .txt。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [multiMotorTCPAPI 源码归档](../../sources/repos/multimotortcpapi.md)（<https://github.com/ti5robot/multiMotorTCPAPI>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：ti5robot/multiMotorTCPAPI@ce2295f，读过 README.md、src/main.cpp、include/socket_service.h、include/socket_client.h、include/Ti5BASIC.h（声明部分）、g++ CMD .txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [钛虎机器人 官方组织](https://github.com/ti5robot)
