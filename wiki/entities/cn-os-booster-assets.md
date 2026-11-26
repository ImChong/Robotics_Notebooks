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
  - ../../sources/repos/booster_assets.md
summary: "加速进化 开源项目 booster_assets（本体模型资产）：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用…"
institutions:
  - booster
---

# booster_assets

## 一句话定义

**booster_assets** 是 [加速进化](https://github.com/BoosterRobotics) 公开的 **本体模型资产** 开源项目：官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

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
| 机构 | 加速进化 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/BoosterRobotics |

### 项目流程图

依据 [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets) 仓库 README 与目录结构：

```mermaid
flowchart LR
  PKG["pip install -e . (pyproject.toml / setup.cfg)"] --> PY["src/booster_assets/motions.py (K1/T1/T2_JOINT_NAMES)"]
  REPO["Booster Assets 仓库资产"] --> ROB["robots/ 本体模型"]
  REPO --> MOT["motions/ 50 Hz 重定向 CSV (7 维根位姿 + 关节弧度)"]
  ROB --> K1["K1/: 22 DoF (URDF/XML) 与 Locomotion (URDF)"]
  ROB --> T1["T1/: 23 DoF / Locomotion (URDF/XML) 与 29 DoF (URDF)"]
  ROB --> T2["T2/: T2_31dof (URDF/XML)"]
  PY --> ALIGN["按关节顺序映射运动轨迹到模型"]
  MOT --> ALIGN
  K1 --> ALIGN
  T1 --> ALIGN
  T2 --> ALIGN
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `booster_assets` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 源码运行时序图

以下时序图基于仓库中 `pyproject.toml` 与 `src/booster_assets/motions.py` 的实际内容编写；本仓库为纯静态资产与关节顺序常量定义包，不包含独立 CLI 或仿真控制循环函数：

```mermaid
sequenceDiagram
  participant Caller as 上游仿真或训练脚本
  participant Pip as pip / setuptools.build_meta (pyproject.toml)
  participant Mod as src/booster_assets/motions.py
  participant Assets as robots/ (URDF, XML) 与 motions/ (CSV)

  Caller->>Pip: python3 -m pip install -e .
  Pip-->>Caller: 按 pyproject.toml 构建元数据安装 booster_assets 可编辑包
  Caller->>Mod: 导入 K1_JOINT_NAMES, T1_JOINT_NAMES, T2_JOINT_NAMES
  Mod-->>Caller: 返回 22 / 23 / 31 个关节名称列表 (无运行时函数调用)
  Caller->>Assets: 按路径读取 robots/K1|T1|T2/*.urdf|.xml 与 motions/K1|T1|T2/*.csv
  Assets-->>Caller: 返回本体描述文件与 50 Hz 重定向轨迹帧 (前 7 列根位姿 + 关节弧度)
```

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [booster_assets 源码归档](../../sources/repos/booster_assets.md)（<https://github.com/BoosterRobotics/booster_assets>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [加速进化 官方组织](https://github.com/BoosterRobotics)
