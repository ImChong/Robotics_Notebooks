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
  - ../../sources/repos/gforcesdkpython.md
summary: "傲意科技 开源项目 gForceSDKPython（SDK/驱动）：gForce 肌电臂环 Python SDK…"
institutions:
  - ohand
---

# gForceSDKPython

## 一句话定义

**gForceSDKPython** 是 [傲意科技](https://github.com/oymotion) 公开的 **SDK/驱动** 开源项目：gForce 肌电臂环 Python SDK

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
| 机构 | 傲意科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/oymotion |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `gForceSDKPython` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`sudo python3 sample.py`（`sample.py` 的 `asyncio.run(main())`，菜单式：扫描 → 连接 → 选择功能）。依赖 `bleak`（见 `requirements.txt`）。

```mermaid
sequenceDiagram
    participant S as main()（sample.py）
    participant G as GForceProfile（gforce.py）
    participant B as bleak（BleakScanner / BleakClient）
    participant A as gForce 臂环（BLE GATT）
    S->>G: GForceProfile()
    S->>G: scan(5, "gForce")
    G->>B: BleakScanner(service_uuids=[SERVICE_GUID]) start / stop
    B-->>S: 扫描结果列表
    S->>G: connect(addr)
    G->>B: BleakClient(addr).connect()
    G->>A: start_notify(CMD_NOTIFY_CHAR f000ffe1…, _onResponse)
    S->>G: setDataNotifSwitch(flags, set_cmd_cb, 1000)
    G->>G: sendCommand：登记 cmdMap 与超时，超过 MTU 时分包
    G->>A: write_gatt_char(CMD 特征值, 命令字节)
    A-->>G: 命令响应 notify
    G->>G: _onResponse：按 cmd 找回调并删除表项
    G-->>S: set_cmd_cb(resp)
    S->>G: startDataNotification(ondata)
    G->>A: start_notify(DATA_NOTIFY_CHAR f000ffe2…, _handleDataNotification)
    loop 数据推送，直到用户按回车
        A-->>G: notify 数据包（四元数 / EMG 原始 / 手势）
        G->>G: _handleDataNotification：NTF_PARTIAL_DATA 拼包，完整后输出
        G-->>S: onData(packet) → ondata(data)
        S->>S: 按 data[0] 类型解析 quaternion / EMG / gesture
    end
    S->>G: stopDataNotification()
    G->>A: stop_notify(DATA)
    S->>G: setDataNotifSwitch(DNF_OFF, set_cmd_cb, 1000)
    S->>G: disconnect()
```

图依据：gForceSDKPython@f20dc45，读过 README.md、sample.py、gforce.py（connect、scan、setDataNotifSwitch、sendCommand、startDataNotification、_handleDataNotification、_onResponse，以及 UUID 常量）；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [gForceSDKPython 源码归档](../../sources/repos/gforcesdkpython.md)（<https://github.com/oymotion/gForceSDKPython>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：oymotion/gForceSDKPython@f20dc45，读过 README.md、sample.py、gforce.py、requirements.txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
