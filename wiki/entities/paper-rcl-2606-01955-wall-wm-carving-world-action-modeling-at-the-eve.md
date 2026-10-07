---
type: entity
tags:
- paper
- awesome-world-action-models-rcl
- rcl-wam-catalog
- repo
- china-embodied-opensource
- open-source
- project
status: complete
updated: 2026-10-06
arxiv: '2606.01955'
code: https://github.com/X-Square-Robot/wall-wm
summary: Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, reporting strong results on its large-scale real-world generalization evaluation.
related:
- paper-rcl-wam-robot-learning-control-survey.md
- ../overview/rcl-awesome-wam-technology-map.md
- ../methods/generative-world-models.md
- ../methods/vla.md
- ../tasks/manipulation.md
- ../tasks/locomotion.md
- ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
- ../entities/humanoid-motion-intelligence.md
- ../queries/china-domestic-opensource-424-coverage.md
sources:
- ../../sources/papers/rcl_awesome_wam_2606_01955_wall-wm-carving-world-action-modeling-at.md
- ../../sources/papers/rcl_awesome_wam_catalog.md
- ../../sources/repos/awesome-world-action-models-rcl.md
- ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
- ../../sources/repos/wall-wm.md
institutions:
- x-square-robot
project_id: rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve
---

# WALL-WM

**WALL-WM: Carving World Action Modeling at the Event Joints** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 531/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, reporting strong results on its large-scale real-world generalization evaluation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, reporting strong results on its large-scale real-world generalization evaluation.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 531/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2606.01955> |
| 代码/项目 | <https://github.com/X-Square-Robot/wall-wm> |
| 子类 / 象限 | 视觉规划与IDM · 记忆与长时序 · 泛化与动作对齐 · Q4 · Dual-system × IDM |

## 核心机制（归纳）

### 策展导读要点

Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, reporting strong results on its large-scale real-world generalization evaluation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.01955) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WALL-WM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 源码运行时序图

**不适用**（现有源码归档只记录公开仓库及项目分类，未保存可辨识的训练/推理脚本或 README 入口；此处保留复现缺口，待核验实现后补图）。

## 项目资源与工程补充

### 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 自变量机器人 |
| 类别 | 世界模型 |
| 官方组织 | https://github.com/X-Square-Robot |

### 工程实践

1. 从官方 GitHub/Gitee 组织检索 `WALL-WM` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

### 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

- [awesome-world-action-models-rcl](paper-rcl-wam-robot-learning-control-survey.md)
- [vla](../methods/vla.md)
- [locomotion](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2606_01955_wall-wm-carving-world-action-modeling-at.md`](../../sources/papers/rcl_awesome_wam_2606_01955_wall-wm-carving-world-action-modeling-at.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2606.01955>

- [WALL-WM 源码归档](../../sources/repos/wall-wm.md)（<https://github.com/X-Square-Robot/WALL-WM>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2606.01955)

- [自变量机器人 官方组织](https://github.com/X-Square-Robot)
