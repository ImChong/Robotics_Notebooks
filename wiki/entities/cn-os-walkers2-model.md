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
  - ../../sources/repos/walkers2-model.md
summary: "优必选 开源项目 WalkerS2-Model（本体模型资产）：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用…"
institutions:
  - ubtech
---

# WalkerS2-Model

## 一句话定义

**WalkerS2-Model** 是 [优必选](https://github.com/UBTECH-Robot) 公开的 **本体模型资产** 开源项目：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

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
| 机构 | 优必选 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/UBTECH-Robot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `WalkerS2-Model` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

本仓库是机器人模型资产（URDF、网格、辅助脚本），不含控制或算法实现；ROS 2 运行入口只有可视化 launch（另有辅助脚本，未画），下图只描述该 launch。

入口：`ros2 launch walker_s2_v1_world_description display.launch.py [rviz:=true urdf_file:=... rviz_config:=...]`

```mermaid
sequenceDiagram
    participant U as 用户
    participant L as launch/display.launch.py
    participant X as check_urdf 与 cat 命令
    participant RSP as robot_state_publisher
    participant JSP as joint_state_publisher_gui
    participant RV as rviz2
    U->>L: ros2 launch walker_s2_v1_world_description display.launch.py [rviz:=true urdf_file:=... rviz_config:=...]
    L->>X: ExecuteProcess(check_urdf urdf_file)，再 Command(cat urdf_file)，默认 urdf/s2_v1_world/s2_v1_world.urdf
    X-->>L: robot_description（URDF 文本；check_urdf 输出到屏幕）
    L->>RSP: 启动节点，参数 robot_description
    L->>JSP: 启动节点
    L->>RV: 启动节点，-d config/rviz/view_robot.rviz（rviz 参数为真时）
    JSP-->>RSP: joint_states（GUI 滑块）
    RSP-->>RV: TF 与 robot_description
```

图依据：WalkerS2-Model@ca2ab98，README.md、walker_s2_v1_world_description/launch/display.launch.py、package.xml；Collected_WalkerS2 的 USD 文件（Isaac Sim 用）及其余两个 URDF 目录未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [WalkerS2-Model 源码归档](../../sources/repos/walkers2-model.md)（<https://github.com/UBTECH-Robot/WalkerS2-Model>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：UBTECH-Robot/WalkerS2-Model@ca2ab98，读过 README.md、walker_s2_v1_world_description/launch/display.launch.py、package.xml；时序图仅覆盖 RViz 可视化 launch，未运行代码。

## 推荐继续阅读

- [优必选 官方组织](https://github.com/UBTECH-Robot)
