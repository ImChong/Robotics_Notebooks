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
  - ../../sources/repos/xgripper.md
summary: "千觉机器人 开源项目 XGripper（SDK/驱动）：Xense 数据采集夹爪 SDK…"
institutions:
  - qianjue
---

# XGripper

## 一句话定义

**XGripper** 是 [千觉机器人](https://github.com/XenseRobotics-AI) 公开的 **SDK/驱动** 开源项目：Xense 数据采集夹爪 SDK

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
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/XenseRobotics-AI、https://github.com/XenseRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `XGripper` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：README 的串口直连用法 `XenseSerialGripper(port="/dev/ttyUSB0")`；或控制台脚本 `xgripper --gripper`（`pyproject.toml` 中 `xgripper = "xgripper.__main__:main"`，运行 `XenseGripperNode`，依赖 `xensesdk.ezros`）。

```mermaid
sequenceDiagram
    participant U as 用户脚本 / 远端调用方
    participant M as xgripper/__main__.py<br/>main()
    participant N as node/xense_gripper_node.py<br/>XenseGripperNode
    participant G as xense_gripper.py<br/>XenseSerialGripper
    participant S as serial_device.py<br/>SerialDevice
    participant MCU as 夹爪 MCU（串口）

    alt 直接串口（README）
        U->>G: XenseSerialGripper(port, baudrate=115200)
    else 控制台脚本 xgripper --gripper
        U->>M: xgripper --gripper
        M->>N: XenseGripperNode()
        N->>N: ezros.init()；Node("gripper_<eth0 MAC>")
        N->>G: XenseGripper.create(port="/dev/ttyS8") 返回串口实现
        N->>N: 建 publisher、5 秒 timer，注册 set_position/set_speed/set_led_color/calibrate 服务
    end
    G->>S: SerialDevice(port, baudrate, timeout)
    G->>G: 启动守护线程 _run_loop 持续 receive()
    U->>G: set_position(40.0, vmax=80, fmax=27)
    G->>G: 校验范围，换算为原始量（位置 raw=85-position）
    G->>S: build_packet(device_id, 8 字节数据域)（头 0x3E + 长度 + CRC16）
    S->>MCU: send 13 字节帧
    U->>G: get_gripper_status()
    G->>S: 发送 GRIPPER_STATUS 查询帧
    MCU-->>S: 13 字节回包
    S-->>G: receive()：校验 CRC，返回 bytes
    G->>G: _run_loop 按 data[3] 解析为状态缓存
    G-->>U: dict（position / velocity / force / temperature）
    opt 节点模式
        N->>G: on_timeout 每 5 秒 get_gripper_status() + get_button_status()
        N-->>U: publisher 发布 gripper_<MAC> 话题
    end
    U->>G: release()（关串口并 join 接收线程）
```

图依据：XenseRobotics-AI/XGripper@130d500，读过 README.md、pyproject.toml、xgripper/__init__.py、xgripper/__main__.py、xgripper/xense_gripper.py（XenseSerialGripper 与 create）、xgripper/serial_device.py、xgripper/node/xense_gripper_node.py；XenseTCPGripper、MasterNode、camera_server、controller 未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [XGripper 源码归档](../../sources/repos/xgripper.md)（<https://github.com/XenseRobotics-AI/XGripper>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：XenseRobotics-AI/XGripper@130d500，读过 README.md、pyproject.toml、xgripper/__init__.py、xgripper/__main__.py、xgripper/xense_gripper.py、xgripper/serial_device.py、xgripper/node/xense_gripper_node.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [千觉机器人 官方组织](https://github.com/XenseRobotics-AI)
