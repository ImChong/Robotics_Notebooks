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
  - ../../sources/repos/leftarmmotionsolver.md
summary: "钛虎机器人 开源项目 LeftArmMotionSolver（工程与工具）：人形左手七轴运动学解算器…"
institutions:
  - ti5robot
---

# LeftArmMotionSolver

## 一句话定义

**LeftArmMotionSolver** 是 [钛虎机器人](https://github.com/ti5robot) 公开的 **工程与工具** 开源项目：人形左手七轴运动学解算器

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
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/ti5robot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `LeftArmMotionSolver` 仓库并核对 README 许可与依赖。
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

入口：`sudo ./move_sov`（由 `src/gcc.sh` 编译：`g++ main.cpp -L./include -lmylibti5 -L./include/can -lmylibscan -lcontrolcan -lfmt -ludev -o move_sov`；README 另给出 2004 版 `libmylibti5_2004` 的编译命令）。开源部分只有 `src/main.cpp` 这一示例程序；其调用的 SDK 函数（`pos_to_move`、`joint_to_move`、`Start`、`Exit` 等）仅有头文件声明，实现在 `usrlib/` 与 `include/can/` 下的闭源 `.so` 中，因此下图只画到 SDK 接口边界，库内部的 IK、运动规划与 CAN 通信未画。

```mermaid
sequenceDiagram
    participant M as src/main.cpp: main()
    participant S as Ti5 SDK 接口 (Ti5BASIC.h / Ti5MOVE.h, 闭源 .so)

    M->>S: query_can() 查询 CAN 设备序列号
    M->>S: ip_address()
    M->>M: signal(SIGINT, signalHandler)
    M->>S: Start() 登录 CAN 设备
    loop 3 轮 (i < 3)
        M->>S: pos_to_move(pos, 237.875, dim=1, absolute=true)
        M->>S: TH.show_crtj / showpointsinfo / show_crtpos
        M->>S: write_value(1, "1.txt", saveJoint)
        M->>S: pos_to_move(pos, 60, dim=1, absolute=false)
        M->>M: pos[0..2] 递增 (+80, +10, +80)
    end
    M->>S: joint_to_move(goal_j2) 7 个目标关节角
    M->>S: mechanical_arm_origin() 回原点
    M->>S: Exit() 登出 CAN 设备
    Note over M,S: SIGINT: signalHandler 调用 brake, inspect_brake, Exit 后 exit
```

图依据：ti5robot/LeftArmMotionSolver@2fee2d1，`README.md`、`src/main.cpp`、`src/gcc.sh`、`g++cmd.txt`、`include/Ti5MOVE.h`、`include/Ti5BASIC.h`、`include/tool.h`、`include/mathfunc.h`（未读取闭源 `.so` 内部实现）。

## 参考来源

- [LeftArmMotionSolver 源码归档](../../sources/repos/leftarmmotionsolver.md)（<https://github.com/ti5robot/LeftArmMotionSolver>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：ti5robot/LeftArmMotionSolver@2fee2d1，读过 README.md、src/main.cpp、src/gcc.sh、g++cmd.txt、include/tool.h，并检索 include/Ti5MOVE.h、include/Ti5BASIC.h、include/mathfunc.h 的声明；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [钛虎机器人 官方组织](https://github.com/ti5robot)
