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
  - ../../sources/repos/mechanical_arm_5_0_python.md
summary: "钛虎机器人 开源项目 mechanical_arm_5_0_python（SDK/驱动）：钛虎机械臂 5.0 Python 控制接口…"
institutions:
  - ti5robot
---

# mechanical_arm_5_0_python

## 一句话定义

**mechanical_arm_5_0_python** 是 [钛虎机器人](https://github.com/ti5robot) 公开的 **SDK/驱动** 开源项目：钛虎机械臂 5.0 Python 控制接口

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

1. 从官方 GitHub/Gitee 组织检索 `mechanical_arm_5_0_python` 仓库并核对 README 许可与依赖。
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

入口（按仓库脚本）：`src/gcc.sh` 生成 `move_sov`，随后 `sudo ./move_sov`。注意：`gcc.sh` 列出的 `communication.cpp`、`mathfunc.cpp`、`Ti5BASIC.cpp`、`Ti5MOVE.cpp`、`Ti5LOGIC.cpp`、`tool.cpp`、`clamping_jaw.cpp`、`example.cpp` 均未入库，仓库内只有头文件与预编译 `.so`，因此按仓库原样无法完成编译。下图只画 `src/main.cpp` 中可见的启动与菜单调度，菜单各 `demo` 函数的实现不在仓库内，未展开。

```mermaid
sequenceDiagram
    participant M as src/main.cpp: main()
    participant S as Ti5 SDK 接口 (头文件 + 闭源 .so, 实现未入库)
    participant K as scanKeyboard() 键盘输入

    M->>S: query_can() 查询 CAN 设备序列号
    M->>S: ip_address()
    M->>M: signal(SIGINT, signalHandler)
    M->>S: login()
    M->>S: mechanical_arm_origin(), sleep(3), show_value(TH.pos / TH.j), brake()
    M->>S: move_to_pos(2)
    loop 菜单循环 while(1)
        M->>K: flag = scanKeyboard()
        alt 按键 1
            M->>S: keyboard_controller()
        else 按键 2
            M->>S: load_program(0, 2, filename)
        else 按键 3
            M->>S: keyboard_save_point()
        else 按键 4 或 5
            M->>S: pour_water() 或 linear_motion()
        else 按键 6 到 b
            M->>S: demo_1 到 demo_6
        else 按键 c
            M->>S: uart485_communication(/dev/ttyUSB0)
        end
        opt flag 为 27 (ESC)
            M->>S: brake()
        end
    end
    Note over M,S: SIGINT: signalHandler 调用 brake, inspect_brake, logout 后 exit
```

图依据：ti5robot/mechanical_arm_5_0_python@43992b8，`README.md`（仅标题）、`src/gcc.sh`、`src/main.cpp`（启动与菜单 switch 段）、`include/Ti5BASIC.h`、`include/tool.h`（声明检索）。

## 参考来源

- [mechanical_arm_5_0_python 源码归档](../../sources/repos/mechanical_arm_5_0_python.md)（<https://github.com/ti5robot/mechanical_arm_5_0_python>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：ti5robot/mechanical_arm_5_0_python@43992b8，读过 README.md（仅标题）、src/gcc.sh、src/main.cpp（启动与菜单段）；检索 include/Ti5BASIC.h、include/tool.h 的声明；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [钛虎机器人 官方组织](https://github.com/ti5robot)
