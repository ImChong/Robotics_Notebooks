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
  - ../../sources/repos/agibot_d1_edu-ultra.md
summary: "智元机器人 开源项目 agibot_D1_Edu-Ultra（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - agibot
---

# agibot_D1_Edu-Ultra

## 一句话定义

**agibot_D1_Edu-Ultra** 是 [智元机器人](https://github.com/AgibotTech) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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

1. 从官方 GitHub/Gitee 组织检索 `agibot_D1_Edu-Ultra` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：高层 `cd demo/zsl-1/cpp && mkdir build && cd build && cmake .. && make -j6 && ./highlevel_demo`，或 `python3 demo/zsl-1/python/examples/highlevel_demo_Interactive.py`；底层 `python3 demo/zsl-1/python/examples/lowlevel_demo.py`。仓库公开的是头文件 `include/zsl-1/*.h`、示例与预编译库 `lib/zsl-1/<arch>/`（`libmc_sdk_zsl_1_*.so` 及 Python 扩展 `mc_sdk_zsl_1_py*.so`）；`docs/architecture.md` 明确不涉及 `/src` 内部实现，故以下只画示例对公开接口的调用。

```mermaid
sequenceDiagram
    participant U as 操作员
    participant H as demo/zsl-1/python/examples/highlevel_demo_Interactive.py
    participant L as demo/zsl-1/python/examples/lowlevel_demo.py MotorControl
    participant S as mc_sdk_zsl_1_py（预编译 SDK 库）
    participant R as 机器狗运动控制
    U->>H: python3 highlevel_demo_Interactive.py
    H->>H: 按 platform.machine 把 lib/zsl-1/arch 加入 sys.path，import mc_sdk_zsl_1_py
    H->>S: HighLevel().initRobot("127.0.0.1", 43988, "127.0.0.1")
    H->>S: standUp() 后 sleep 3 秒
    S->>R: 经 SDK 通信链路下发（协议在闭源库内）
    loop 菜单循环
        U->>H: 输入编号
        H->>S: move(vx, vy, yaw) / jump / frontJump / backflip / shakeHand / attitudeControl / lieDown
        H->>S: 移动类动作 sleep 后 move(0, 0, 0)，再 standUp 回到稳定状态
    end
    U->>L: python3 lowlevel_demo.py
    L->>S: LowLevel().initRobot(local_ip, local_port, dog_ip)
    loop 每 2ms
        L->>S: getMotorState() 与 haveMotorData()
        S-->>L: 关节 q_abad / q_hip / q_knee
        L->>L: 线性插值，两阶段目标姿态
        L->>S: sendMotorCmd(MotorCommand q_des kp kd)
        S->>R: 电机指令
    end
```

图依据：AgibotTech/agibot_D1_Edu-Ultra@b5fe0a8，读过 `README.md`、`docs/architecture.md`、`demo/zsl-1/python/examples/highlevel_demo_Interactive.py`、`demo/zsl-1/python/examples/lowlevel_demo.py`（前 90 行）、`demo/zsl-1/cpp/highlevel_demo.cpp`（`main` 与按键分支）、`include/zsl-1/highlevel.h` 与 `lowlevel.h`（接口签名）。`zsl-1w` 示例、`switch_control.cpp`、`video.py` 与 `docs/` 其余文档未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [agibot_D1_Edu-Ultra 源码归档](../../sources/repos/agibot_d1_edu-ultra.md)（<https://github.com/AgibotTech/agibot_D1_Edu-Ultra>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：AgibotTech/agibot_D1_Edu-Ultra@b5fe0a8，读过 README.md、docs/architecture.md、demo/zsl-1/python/examples/highlevel_demo_Interactive.py、lowlevel_demo.py、demo/zsl-1/cpp/highlevel_demo.cpp、include/zsl-1/highlevel.h、lowlevel.h（接口签名）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智元机器人 官方组织](https://github.com/AgibotTech)
