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
  - ../../sources/repos/agillink_omnihand_sdk.md
summary: "智元机器人 开源项目 agillink_omnihand_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - agibot
---

# agillink_omnihand_sdk

## 一句话定义

**agillink_omnihand_sdk** 是 [智元机器人](https://github.com/AgibotTech) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 智元机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/AgibotTech、https://github.com/AimRT、https://github.com/OpenDriveLab |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `agillink_omnihand_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：在 `linux/x64/` 目录下执行 `PYTHONPATH=python uvicorn omnihand_server.app.main:app --reload --port 8000`（REST/WS 服务路径）。另有直接调用 SDK 的 demo：`python python/demo/omnihand_2025/demo_set_angle.py -d zlgcan`，其调用链为 demo 直接 `from omnihand import OmniHand2025`，不经过服务层，本图未展开。

说明：仓库内 `omnihand` 原生模块仅以 `python/*.whl` 与 `cpp/include`、`cpp/lib` 形式提供，C++ 核心实现源码不在仓库内，图中 `HandService` 段仅记录 Python 侧可见的调用边界。

```mermaid
sequenceDiagram
    participant C as 客户端 REST 或 WS
    participant A as omnihand_server/app/main.py<br/>FastAPI app
    participant H as omnihand_server/routers/hand_router.py<br/>前缀 /v1/hands
    participant K as omnihand_server/core/sdk.py<br/>产品类解析
    participant V as omnihand_server/core/service.py<br/>get_service
    participant N as omnihand.omnihand_core.HandService<br/>C++ pybind 二进制
    participant D as CAN 或串口设备
    Note over A: uvicorn 导入 app 时创建 FastAPI 并注册 CORS 与异常处理
    C->>H: POST /v1/hands/ HandCreateRequest
    H->>K: get_hand_class 与 get_product_type 按 hand_type 解析
    H->>V: get_service().create_hand(spec)
    V->>N: 首次调用时 import omnihand_core 并实例化 HandService
    N->>D: 按 conn_method 如 zlgcan hcan socketcan rs485 打开设备并初始化
    D-->>N: 句柄与状态
    N-->>V: 结果
    V-->>H: result
    H-->>C: 201 JSON
    C->>H: POST /v1/hands/ID/methods 携带 method 与 body
    H->>V: call_method(hand_id, method, body)
    V->>N: 转发到已建立的会话
    N->>D: 下发指令
    D-->>N: 反馈
    N-->>V: 返回值
    V-->>H: 返回值
    H-->>C: JSON
    C->>A: WebSocket /ws 订阅流
    A->>V: run_in_executor 调用 read_stream(hand_id, stream)
    V->>N: 读取数据流
    N-->>A: 推送到 WebSocket
    Note over A: 进程退出时 lifespan 调用 shutdown 销毁 HandService
```

图依据：AgibotTech/agillink_omnihand_sdk@1b552ea，读过 linux/x64/python/omnihand_server/app/main.py、routers/hand_router.py、core/service.py、core/sdk.py（函数列表）、linux/x64/python/omnihand_server/readme.md、linux/x64/python/demo/omnihand_2025/demo_set_angle.py、doc/en/QUICK_START.md；routers/ws.py 仅读取 read_stream 调用行。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [agillink_omnihand_sdk 源码归档](../../sources/repos/agillink_omnihand_sdk.md)（<https://github.com/AgibotTech/agillink_omnihand_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：AgibotTech/agillink_omnihand_sdk@1b552ea，读过 linux/x64/python/omnihand_server/app/main.py、routers/hand_router.py、core/service.py、core/sdk.py、readme.md、linux/x64/python/demo/omnihand_2025/demo_set_angle.py、doc/en/QUICK_START.md；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智元机器人 官方组织](https://github.com/AgibotTech)
