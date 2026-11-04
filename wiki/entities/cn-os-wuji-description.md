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
  - ../../sources/repos/wuji-description.md
summary: "舞肌科技 开源项目 wuji-description（本体模型资产）：提供Wuji Hand及相关设备的URDF、MJCF、MJX和USD模型资产，连接ROS、MuJoCo、JAX和Isaac仿真流程。…"
institutions:
  - wuji-robotics
---

# wuji-description

## 一句话定义

**wuji-description** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **本体模型资产** 开源项目：提供Wuji Hand及相关设备的URDF、MJCF、MJX和USD模型资产，连接ROS、MuJoCo、JAX和Isaac仿真流程。

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
| 机构 | 舞肌科技 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wuji-description` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch wuji_description display.launch.py hand:=left`（`hand/body/launch/display.launch.py`，ROS2 包 `wuji_description` 需先 colcon 构建；`hand` 可选 `left` 或 `right`）。仓库以 URDF、MJCF、USD 模型资产为主，此处仅画唯一的可运行入口（可视化启动文件）。

```mermaid
sequenceDiagram
    participant U as 用户 shell
    participant L as launch/display.launch.py generate_launch_description
    participant LS as launch_setup（OpaqueFunction）
    participant F as share/wuji_description urdf 下 hand-ros.urdf
    participant RSP as robot_state_publisher
    participant JG as joint_state_publisher_gui
    participant RV as rviz2 与 rviz/hand.rviz
    U->>L: ros2 launch 启动 display.launch.py
    L->>L: 声明参数 hand（left 或 right）、use_sim_time、gui
    L->>LS: 运行 launch_setup 并解析 hand 的实际取值
    LS->>F: 读取 {hand}-ros.urdf 文本（package:// 路径）
    LS->>RSP: 创建节点，robot_description 参数为 URDF 文本
    LS->>JG: 创建 joint_state_publisher_gui 节点
    LS->>RV: 创建 rviz2 节点，参数 -d 指向 rviz/{hand}.rviz
    JG-->>RSP: 滑块关节角经 joint_states 话题送出
    RSP-->>RV: 发布 TF 供 RViz 渲染模型
```

图依据：wuji-description@5c4dfa1，README.md（仓库结构与用途）、hand/body/launch/display.launch.py（完整读过）、hand/body/package.xml 与 CMakeLists.txt（仅 grep 包名与依赖）。joint_states 与 TF 话题是 ROS 标准约定，仓库内未重新定义。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wuji-description 源码归档](../../sources/repos/wuji-description.md)（<https://github.com/wuji-technology/wuji-description>）

- 源码核对（2026-11-02）：wuji-technology/wuji-description@5c4dfa1，读过 README.md、hand/body/launch/display.launch.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
