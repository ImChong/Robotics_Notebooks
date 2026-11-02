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
  - ../../sources/repos/rm_api2.md
summary: "睿尔曼智能 开源项目 RM_API2（SDK/驱动）：提供睿尔曼机器人新一代控制API，使上层应用读取状态并发送机械臂控制命令。…"
institutions:
  - realman
---

# RM_API2

## 一句话定义

**RM_API2** 是 [睿尔曼智能](https://github.com/RealManRobot) 公开的 **SDK/驱动** 开源项目：提供睿尔曼机器人新一代控制API，使上层应用读取状态并发送机械臂控制命令。

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
| 机构 | 睿尔曼智能 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/RealManRobot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RM_API2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python ./src/main.py`（在 `Demo/RMDemo_Python/RMDemo_SimpleProcess` 目录下；`main()` 位于 `src/core/demo_simple_process.py`）。本仓库是预编译库（`libapi_c.so` / `api_c.dll`）加头文件与多语言 Demo，库内部实现与线协议不在仓库源码中，图只画到库边界。

```mermaid
sequenceDiagram
    participant M as src/main.py
    participant D as demo_simple_process.main()<br/>RobotArmController
    participant R as Robotic_Arm.rm_robot_interface.RoboticArm
    participant W as rm_ctypes_wrap(ctypes)
    participant L as libapi_c.so(预编译，闭源)
    participant C as 机械臂控制器(192.168.1.18:8080)
    M->>D: main()
    D->>R: RoboticArm(rm_thread_mode_e(2))
    R->>W: rm_init(mode) 并打印 rm_api_version()
    D->>R: rm_create_robot_arm("192.168.1.18", 8080, 3)
    R->>W: rm_set_log_call_back / rm_create_robot_arm / rm_get_robot_info
    W->>L: 调用 C 接口(ctypes cdecl)
    L-->>C: 建立连接(库内部实现未读)
    R-->>D: handle(id==-1 则退出)
    D->>R: rm_get_arm_software_info()
    D->>R: rm_get_robot_info() 取 arm_model
    Note over D: 按 arm_model 从 arm_models_to_points 取示教点
    D->>R: rm_movej(joint, v=20, block=1)
    D->>R: rm_movej_p(points[1]) 、 rm_movel(points[2]) 、 rm_movej_p(points[3])
    D->>R: rm_movec(points[4], points[5], loop=2)
    R->>W: rm_movej / rm_movej_p / rm_movel / rm_movec
    W->>L: 运动指令(block=1 阻塞到返回码)
    L-->>D: 返回码 0 成功，非 0 打印错误码
    D->>R: rm_delete_robot_arm()
```

图依据：RealManRobot/RM_API2@a810b9f，读过 README.md、Python/Robotic_Arm/rm_robot_interface.py（RoboticArm、rm_create_robot_arm、rm_delete_robot_arm、rm_movej 等）、rm_ctypes_wrap.py（库加载部分）、Demo/RMDemo_Python/RMDemo_SimpleProcess/readme.md、src/main.py、src/core/demo_simple_process.py；libapi_c.so 内部未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RM_API2 源码归档](../../sources/repos/rm_api2.md)（<https://github.com/RealManRobot/RM_API2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：RealManRobot/RM_API2@a810b9f，读过 README.md、Python/Robotic_Arm/rm_robot_interface.py 与 rm_ctypes_wrap.py（部分）、Demo/RMDemo_Python/RMDemo_SimpleProcess 的 readme.md、src/main.py、src/core/demo_simple_process.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [睿尔曼智能 官方组织](https://github.com/RealManRobot)
