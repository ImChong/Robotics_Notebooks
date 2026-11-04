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
  - ../../sources/repos/mobile_aloha_sim.md
summary: "松灵机器人 开源项目 mobile_aloha_sim（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - agilex
---

# mobile_aloha_sim

## 一句话定义

**mobile_aloha_sim** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

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
| 机构 | 松灵机器人 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `mobile_aloha_sim` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`roslaunch aloha_new_description display_aloha_tracer2_dabai.launch`（README 写作「执行 display_aloha_tracer2_dabai.launch」，roslaunch 为 ROS1 标准命令）

该 launch 只做模型展示（URDF + rviz），不含控制或仿真逻辑。

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant L as display_aloha_tracer2_dabai.launch
    participant P as ROS 参数服务器
    participant JS as joint_state_publisher_gui 节点
    participant RS as robot_state_publisher 节点
    participant RV as rviz 节点
    U->>L: roslaunch 执行 launch 文件
    L->>P: param robot_description 由 textfile 读入 urdf/aloha_tracer2_dabai_dark.urdf
    L->>JS: 启动 joint_state_publisher_gui（GUI 滑条调关节角）
    L->>RS: 启动 robot_state_publisher
    L->>RV: 启动 rviz，-d rviz/aloha.rviz
    JS-->>RS: 关节状态（ROS 约定话题 joint_states，非本仓库代码）
    RS->>P: 读取 robot_description 并发布坐标变换
    RS-->>RV: 模型与 TF 显示
```

图依据：agilexrobotics/mobile_aloha_sim@f799ae0，读过 README.MD、aloha_new_description/README.MD、aloha_new_description/launch/display_aloha_tracer2_dabai.launch、aloha_new_description/launch/display_aloha_tracer2_d435.launch、aloha_new_description/CMakeLists.txt、aloha_new_description/package.xml；urdf 仅核对存在、未读内容。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [mobile_aloha_sim 源码归档](../../sources/repos/mobile_aloha_sim.md)（<https://github.com/agilexrobotics/mobile_aloha_sim>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：agilexrobotics/mobile_aloha_sim@f799ae0，读过 README.MD、aloha_new_description/README.MD、aloha_new_description/launch/display_aloha_tracer2_dabai.launch、aloha_new_description/launch/display_aloha_tracer2_d435.launch、aloha_new_description/CMakeLists.txt、aloha_new_description/package.xml；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
