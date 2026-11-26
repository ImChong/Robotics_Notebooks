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
  - ../../sources/repos/lightwheel-ycb.md
summary: "光轮智能 开源项目 Lightwheel-YCB（仿真环境）：提供刚体、关节体和柔性物体的仿真就绪YCB资产，并同时提供MJCF与USD格式。…"
institutions:
  - lightwheel
---

# Lightwheel-YCB

## 一句话定义

**Lightwheel-YCB** 是 [光轮智能](https://github.com/LightwheelAI) 公开的 **仿真环境** 开源项目：提供刚体、关节体和柔性物体的仿真就绪YCB资产，并同时提供MJCF与USD格式。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 光轮智能 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/LightwheelAI |

### 项目流程图

依据 [LightwheelAI/Lightwheel-YCB](https://github.com/LightwheelAI/Lightwheel-YCB) 仓库 README 描述的资产增强与标准化管线：

```mermaid
flowchart LR
  IN["YCB Benchmarks (106 物体: OBJ/PLY/STL) + 新增 19 个方块"] --> VIS["视觉增强 (网格修复与 PBR 材质)"]
  IN --> PHY["物理参数补充 (质量 / 摩擦 / 密度 / 阻尼 / 刚度 / 惯量)"]
  IN --> COL["碰撞体重建与动态变体 (刚体 / 16 铰接 / 2 柔性)"]
  VIS --> QA["Isaac Lab 与 MuJoCo 遥操作质量验证"]
  PHY --> QA
  COL --> QA
  QA --> USD["125 个 USD 资产 (Isaac Sim / Isaac Lab)"]
  QA --> MJCF["123 个 MJCF 资产 (MuJoCo; 2 个柔性仅 USD)"]
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Lightwheel-YCB` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

- [机器人仿真环境集成](../methods/robot-simulation-integration.md)：了解模型、控制与仿真接口。
- [机器人软件系统集成](../methods/robot-software-integration.md)：对照仿真与软件模块的接口。

## 参考来源

- [Lightwheel-YCB 源码归档](../../sources/repos/lightwheel-ycb.md)（<https://github.com/LightwheelAI/Lightwheel-YCB>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [光轮智能 官方组织](https://github.com/LightwheelAI)
