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
  - ../../sources/repos/ht_robot_urdf.md
summary: "高擎机电 开源项目 robot_urdf（本体模型资产）：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用…"
institutions:
  - hightorque
---

# robot_urdf

## 一句话定义

**robot_urdf** 是 [高擎机电](https://github.com/HighTorque-Robotics) 公开的 **本体模型资产** 开源项目：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

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
| 机构 | 高擎机电 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/HighTorque-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `robot_urdf` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`roslaunch hi_25dof display.launch`（`gazebo.launch` 为仿真加载流程，本图未展开）

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant L as hi_25dof/launch/display.launch
    participant P as ROS 参数 robot_description
    participant JG as joint_state_publisher_gui
    participant RSP as robot_state_publisher
    participant R as rviz (launch/urdf.rviz)
    U->>L: roslaunch hi_25dof display.launch
    L->>P: param robot_description 读取 urdf/hi_25dof.urdf（textfile）
    L->>JG: node joint_state_publisher_gui
    L->>RSP: node robot_state_publisher
    L->>R: node rviz，-d launch/urdf.rviz
    JG-->>RSP: /joint_states 关节角（GUI 滑条输入）
    P-->>RSP: 模型描述（URDF 连杆与关节树）
    RSP-->>R: /tf 坐标变换
    R-->>U: 显示 hi_25dof 模型
```

图依据：HighTorque-Robotics/HT_Robot_URDF@5675b9a，`hi_25dof/launch/display.launch`、`hi_25dof/launch/gazebo.launch`、`hi_25dof/CMakeLists.txt`、`README.md`。`/joint_states` 与 `/tf` 的话题约定属于 ROS 标准包行为，仓库内未读对应源码，未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [robot_urdf 源码归档](../../sources/repos/ht_robot_urdf.md)（<https://github.com/HighTorque-Robotics/HT_Robot_URDF>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：HighTorque-Robotics/HT_Robot_URDF@5675b9a，读过 `README.md`、`hi_25dof/launch/display.launch`、`hi_25dof/launch/gazebo.launch`、`hi_25dof/CMakeLists.txt`；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [高擎机电 官方组织](https://github.com/HighTorque-Robotics)
