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
  - ../../sources/repos/seertcptest.md
summary: "仙工智能 开源项目 SeerTCPTest（工程与工具）：提供Robokit NetProtocol TCP API的Qt源码与测试工具，可发送控制请求并检查移动机器人响应和日志。…"
institutions:
  - seer
---

# SeerTCPTest

## 一句话定义

**SeerTCPTest** 是 [仙工智能](https://github.com/seer-robotics) 公开的 **工程与工具** 开源项目：提供Robokit NetProtocol TCP API的Qt源码与测试工具，可发送控制请求并检查移动机器人响应和日志。

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
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/seer-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `SeerTCPTest` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`main.cpp` 的 `main`（Qt 工程 `SeerTools.pro`，GUI 程序：`QApplication` → `SCTcpToolWidget` → `show()`；README 指向的 `win_x64_bin/SeerTools.exe` 在仓库中不存在，仓库内仅有 `win_x64_bin_Qt5/`、`win_x64_bin_Qt6/` 两个预编译目录）

```mermaid
sequenceDiagram
    participant User as 操作者
    participant UI as SCTcpToolWidget（界面）
    participant Tcp as SCStatusTcp（协议层）
    participant Sock as QTcpSocket
    participant Bot as 机器人 Robokit TCP 端口
    User->>UI: 点击连接
    UI->>Tcp: connectHost(ip, port)
    Tcp->>Sock: new QTcpSocket，setProxy(NoProxy)，连接 readyRead / stateChanged / error
    Tcp->>Sock: connectToHost(ip, port, IPv4)
    Sock->>Bot: TCP 建连
    Sock-->>UI: stateChanged → SCTcpToolWidget::stateChanged
    User->>UI: 点击发送
    UI->>UI: on_pushButton_send_clicked：读取 API 编号、数据区（若为文件则读入二进制）、JSON 区、序号、byte15
    UI->>Tcp: writeTcpData(cmd, json, data, number, byte15)
    Tcp->>Tcp: 组包：16 字节头（大端：长度 / 类型 / 序号）+ JSON + 数据
    Tcp->>Sock: 写入请求报文
    opt 指令超时已开启（_timeOut ≠ 0）
        Tcp->>Sock: waitForBytesWritten(_timeOut)
        Tcp->>Sock: waitForReadyRead(_timeOut)，超时则返回 false，界面打印 Send error
    end
    Sock->>Bot: 请求报文
    Bot-->>Sock: 响应报文
    Sock-->>Tcp: readyRead → receiveTcpReadyRead
    Tcp->>Tcp: 拼接 _lastMessage，解析头部，长度不足则缓存等待下一包
    Tcp-->>UI: sigChangedText(true, revCommand, data, hex, number, msTime)
    UI->>UI: slotChangedText：显示返回内容，可另存为文件
    opt 勾选定时发送
        UI->>UI: QTimer.timeout → on_pushButton_send_clicked
    end
    Note over UI,Tcp: 开启超时时，writeTcpData 在界面线程内阻塞等待，期间界面不响应
```

图依据：SeerTCPTest@47ea4fe，`main.cpp`、`SCTcpToolWidget.cpp`（构造、连接、on_pushButton_send_clicked、slotChangedText、定时发送）、`SCStatusTcp.cpp`（connectHost、writeTcpData、receiveTcpReadyRead）、`README.md`；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [SeerTCPTest 源码归档](../../sources/repos/seertcptest.md)（<https://github.com/seer-robotics/SeerTCPTest>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：seer-robotics/SeerTCPTest@47ea4fe，读过 README.md、main.cpp、SCTcpToolWidget.cpp（连接、发送、返回、定时发送段）、SCStatusTcp.cpp（connectHost、writeTcpData、receiveTcpReadyRead 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [仙工智能 官方组织](https://github.com/seer-robotics)
