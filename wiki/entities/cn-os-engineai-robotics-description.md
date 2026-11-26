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
  - ../../sources/repos/engineai_robotics_description.md
summary: "众擎机器人 开源项目 engineai_robotics_description（本体模型资产）：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用…"
institutions:
  - engineai
---

# engineai_robotics_description

## 一句话定义

**engineai_robotics_description** 是 [众擎机器人](https://github.com/engineai-robotics) 公开的 **本体模型资产** 开源项目：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

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
| 机构 | 众擎机器人 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/engineai-robotics |

### 项目流程图

依据 [engineai-robotics/engineai_robotics_description](https://github.com/engineai-robotics/engineai_robotics_description) 仓库 README 与目录结构：

```mermaid
flowchart LR
  REPO["engineai_robotics_description 资产仓"] --> PM["PM01: pm01_edu/ (v2) 与 pm01_lidar/ (v1)"]
  REPO --> T8["T800: t800/ (v1.1)"]
  REPO --> T8P["T800 Pro: t800pro/t800pro_25dof (URDF + USD) 与 t800pro_43dof (URDF)"]
  PM --> SUB["型号子目录约定: urdf/ 与 xml/ (MuJoCo) 与 meshes/ 与 texture/ 与 usd/ (Isaac Lab)"]
  T8 --> SUB
  T8P --> SUB
  SUB -->|"部分型号待同步"| DOWN["engineai_rl_lab / engineai_robotics_native_sdk 等下游训练与部署仓"]
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `engineai_robotics_description` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

- [机器人模型描述与资产接入](../methods/robot-model-description-workflow.md)：了解连杆、关节和模型资产。
- [机器人软件系统集成](../methods/robot-software-integration.md)：对照模型在软件工具链中的接入。

## 参考来源

- [engineai_robotics_description 源码归档](../../sources/repos/engineai_robotics_description.md)（<https://github.com/engineai-robotics/engineai_robotics_description>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [众擎机器人 官方组织](https://github.com/engineai-robotics)
