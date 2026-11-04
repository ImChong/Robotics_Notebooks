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
  - ../../sources/repos/pymycobot.md
summary: "大象机器人 开源项目 pymycobot（工程与工具）：提供大象机器人多类机械臂和双臂产品的统一Python API，用于连接、状态读取、关节与末端控制及外设集成。…"
institutions:
  - elephant-robotics
---

# pymycobot

## 一句话定义

**pymycobot** 是 [大象机器人](https://github.com/elephantrobotics) 公开的 **工程与工具** 开源项目：提供大象机器人多类机械臂和双臂产品的统一Python API，用于连接、状态读取、关节与末端控制及外设集成。

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
| 机构 | 大象机器人 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/elephantrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `pymycobot` 仓库并核对 README 许可与依赖。
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

入口：用户脚本中实例化 `MyCobot280("/dev/ttyUSB0", "115200")`（串口）或 `MyCobot280Socket(ip, 9000)`（网络），再调用 `get_angles()` 等接口。下图以 `get_angles()` 为例，两条通道共用同一组命令生成与解析逻辑。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户脚本
    participant M as MyCobot280（mycobot280.py）
    participant S as MyCobot280Socket（mycobot280socket.py）
    participant G as CommandGenerator.get_angles（generate.py）
    participant DP as DataProcessor（common.py：_mesg / _flatten / _process_received）
    participant SER as pyserial Serial（串口）
    participant SK as socket（TCP）
    participant HW as 机械臂固件
    U->>M: MyCobot280(port, baudrate)，内部 serial.Serial().open()
    U->>S: MyCobot280Socket(ip, netport)，内部 connect_socket()
    U->>M: get_angles()
    M->>G: get_angles() → _mesg(ProtocolCode.GET_ANGLES, has_reply=True)
    G->>DP: DataProcessor._mesg(genre, *args, has_reply, _async) 组帧
    DP-->>M: real_command，has_reply，_async
    alt 串口通道
        M->>M: thread_lock 为真时持有 self.lock，调用 _res
        loop 最多 3 次，直到收到非空回包
            M->>SER: _write(_flatten(real_command)) → serial.write
            M->>SER: _read(genre) → inWaiting / read
        end
        M->>DP: _process_received(data, genre)
        DP-->>U: 角度列表
    else 网络通道
        U->>S: get_angles() 经 _mesg 组帧（同上）
        S->>SK: _write(_flatten(real_command), "socket") → sock.sendall
        S->>SK: _read(genre, method="socket") → sock.recv(1024)
        S->>S: 校验回包 data[3] 与 genre 一致，失败则重试
        S-->>U: 角度列表
    end
    Note over M,HW: 由 pymycobot 经串口 / 网络把协议帧发给机械臂固件，固件内部逻辑不在仓库内
```

图依据：elephantrobotics/pymycobot@44d7fb1，读过 `README.md`、`pymycobot/__init__.py`、`pymycobot/mycobot.py`、`pymycobot/mycobot280.py`、`pymycobot/mycobot280socket.py`、`pymycobot/mycobotsocket.py`（部分）、`pymycobot/generate.py`（部分）、`pymycobot/common.py`（部分）；未运行代码。

## 参考来源

- [pymycobot 源码归档](../../sources/repos/pymycobot.md)（<https://github.com/elephantrobotics/pymycobot>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：elephantrobotics/pymycobot@44d7fb1，读过 `README.md`、`pymycobot/__init__.py`、`pymycobot/mycobot.py`、`pymycobot/mycobot280.py`、`pymycobot/mycobot280socket.py`、`pymycobot/mycobotsocket.py`（部分）、`pymycobot/generate.py`（部分）、`pymycobot/common.py`（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [大象机器人 官方组织](https://github.com/elephantrobotics)
