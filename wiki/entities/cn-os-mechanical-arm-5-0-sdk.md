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
  - ../../sources/repos/mechanical_arm_5_0_sdk.md
summary: "钛虎机器人 开源项目 mechanical_arm_5_0_SDK（SDK/驱动）：钛虎机械臂 5.0 控制 SDK…"
institutions:
  - ti5robot
---

# mechanical_arm_5_0_SDK

## 一句话定义

**mechanical_arm_5_0_SDK** 是 [钛虎机器人](https://github.com/ti5robot) 公开的 **SDK/驱动** 开源项目：钛虎机械臂 5.0 控制 SDK

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

1. 从官方 GitHub/Gitee 组织检索 `mechanical_arm_5_0_SDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`sudo ./move_sov`（由 `code/src/gcc.sh` 编译 `code/src/main.cpp` 得到，见 README「SDK运行」）

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant M as main.cpp 的 main()
    participant B as Ti5BASIC.h 声明的函数
    participant V as Ti5MOVE.h 声明的函数
    Note over B,V: 仓库只有头文件与预编译 .so，这两组函数的实现源码未提供
    U->>M: sudo ./move_sov
    M->>B: query_can() 查询 CAN 设备序列号
    alt 没有设备
        M-->>U: 提示未找到 USB 设备并 exit(0)
    else 找到设备
        M->>B: ip_address()
        M->>M: signal(SIGINT, signalHandler)
        M->>B: login() 登录 CAN 设备
        M->>V: mechanical_arm_origin() 回零
        M->>V: brake()
        loop 循环 5 次
            M->>V: move_to_joint() 按 TH.j 关节角运动
            M->>V: mechanical_arm_origin()
            M->>V: move_to_pos() 按 TH.pos 位姿运动，x 依次 150 250 350 450，段间 sleep(3)
            M->>V: mechanical_arm_origin()
        end
        M->>B: logout()
    end
    Note over M: Ctrl+C 触发 signalHandler: brake() → inspect_brake() → logout() → exit
```

图依据：ti5robot/mechanical_arm_5_0_SDK@750bdd6，读过 code/src/main.cpp、code/src/gcc.sh、code/include/Ti5BASIC.h、code/include/Ti5MOVE.h、code/include/tool.h、机械臂使用/SDK运行.md。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [mechanical_arm_5_0_SDK 源码归档](../../sources/repos/mechanical_arm_5_0_sdk.md)（<https://github.com/ti5robot/mechanical_arm_5_0_SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：ti5robot/mechanical_arm_5_0_SDK@750bdd6，读过 README.md、机械臂使用/SDK运行.md、code/src/main.cpp、code/src/gcc.sh、code/src_pyt/test.py、code/include/Ti5BASIC.h、code/include/Ti5MOVE.h、code/include/tool.h；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [钛虎机器人 官方组织](https://github.com/ti5robot)
