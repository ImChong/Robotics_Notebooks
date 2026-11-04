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
  - ../../sources/repos/flexiv_rdk.md
summary: "非夕科技 开源项目 flexiv_rdk（SDK/驱动）：提供非夕机器人C++与Python开发接口，使控制、状态、力控和应用程序接入真实机器人。…"
institutions:
  - flexiv
---

# flexiv_rdk

## 一句话定义

**flexiv_rdk** 是 [非夕科技](https://github.com/flexivrobotics) 公开的 **SDK/驱动** 开源项目：提供非夕机器人C++与Python开发接口，使控制、状态、力控和应用程序接入真实机器人。

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
| 机构 | 非夕科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/flexivrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `flexiv_rdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（需先按官方 First Time Setup 完成准备，连接的是真实或已配置的 Flexiv 机器人）：`python3 example_py/basics3_primitive_execution.py Enlight-L-123456`

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户 终端
    participant E as basics3_primitive_execution.py main
    participant R as flexivrdk.Robot Python 封装
    participant C as 机器人控制器 闭源库经网络
    participant T as utility.py 工具函数
    U->>E: python3 basics3_primitive_execution.py robot_sn
    E->>R: flexivrdk.Robot(robot_sn) 按序列号连接
    E->>R: fault() 若有故障则 ClearFault()
    E->>R: ServoOn() 使能伺服
    loop 直到 operational() 为真
        E->>R: operational()
    end
    E->>R: info().single_arm_groups 获取单臂关节组
    E->>R: Home() 回到 home 位姿
    E->>R: SwitchMode(NRT_PRIMITIVE_EXECUTION)
    E->>R: ExecutePrimitive 对各关节组下发 MoveJ (target 与 waypoints)
    R->>C: 下发 MoveJ 基元
    loop 直到各组 reachedTarget 为真
        E->>R: primitive_states()
        E->>T: primitive_state_true_for_groups(states, "reachedTarget")
    end
    E->>R: ExecutePrimitive 对各关节组下发 MoveL (世界系目标位姿与 waypoints)
    R->>C: 下发 MoveL 基元
    loop 直到 reachedTarget 为真
        E->>R: primitive_states()
    end
    E->>T: quat2eulerZYX(target_quat, degree=True)
    E->>R: ExecutePrimitive 下发 MoveL (TRAJ START 参考系, 仅旋转)
    R->>C: 下发 MoveL 基元
    loop 直到 reachedTarget 为真
        E->>R: primitive_states()
    end
    E->>R: Stop() 停止并回到 IDLE
```

图依据：flexivrobotics/flexiv_rdk@159d41f，README.md、example_py/basics3_primitive_execution.py、example_py/utility.py、example_py/basics2_clear_fault.py（检索）。库内部（lib/ 下预编译静态库）为闭源，图中 `Robot` 与控制器之间的调用未展开，未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [flexiv_rdk 源码归档](../../sources/repos/flexiv_rdk.md)（<https://github.com/flexivrobotics/flexiv_rdk>）

- 源码核对（2026-11-02）：flexivrobotics/flexiv_rdk@159d41f，读过 README.md、example_py/basics3_primitive_execution.py、example_py/utility.py，检索过 example_py/basics2_clear_fault.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [非夕科技 官方组织](https://github.com/flexivrobotics)
