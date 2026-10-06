---
type: entity
tags:
- physical-ai
- awesome-physical-ai
- dataset
- paper
- awesome-world-action-models-rcl
- rcl-wam-catalog
status: complete
updated: 2026-10-06
code: https://github.com/rail-berkeley/bridge_data_v2
summary: Diverse manipulation behaviours designed to support broad generalisation.
related:
- ../entities/awesome-physical-ai-natnew.md
- ../overview/awesome-physical-ai-technology-map.md
- ../methods/vla.md
- ../tasks/manipulation.md
- paper-rcl-wam-robot-learning-control-survey.md
- ../overview/rcl-awesome-wam-technology-map.md
- ../methods/generative-world-models.md
- ../tasks/locomotion.md
sources:
- ../../sources/repos/pai_awesome_dataset_085_bridgedata-v2.md
- ../../sources/repos/awesome-physical-ai-union-catalog.md
- ../../sources/repos/awesome-physical-ai-natnew.md
- ../../sources/repos/awesome-physical-ai-aichr.md
- ../../sources/papers/rcl_awesome_wam_ref_d0a7d699e0efc759ae64_bridgedata-v2-a-dataset-for-robot-learni.md
- ../../sources/papers/rcl_awesome_wam_catalog.md
- ../../sources/repos/awesome-world-action-models-rcl.md
project_id: 085-bridgedatav2
venue: CoRL 2023
---

# BridgeData V2

**BridgeData V2** 收录于 awesome-physical-ai（natnew、aichr）**第 085/384** 条，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与官方入口，细节以官方文档 / 原文为准。

## 一句话定义

Diverse manipulation behaviours designed to support broad generalisation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 具身/物理智能策展主题 |
| OXE | Open X-Embodiment | 跨本体轨迹语料参照 |
| IL | Imitation Learning | 演示数据驱动的模仿学习 |

| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- Diverse manipulation behaviours designed to support broad generalisation.
- 想横向对照同一分组的其他工作，可以从 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md) 逐条展开。
- 两份清单合并去重：同一条目只有这一页，出处见下方「参考来源」。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 085/384 |
| 分组 | Datasets |
| 来源清单 | natnew、aichr |
| 主链接 | <https://rail-berkeley.github.io/bridgedata/> |
| 代码/仓库 | <https://github.com/rail-berkeley/bridge_data_v2> |

## 核心原理

Diverse manipulation behaviours designed to support broad generalisation.

该条目在 Physical AI 清单中的角色是 **dataset**，分组 **Datasets**。本页只给出清单里的问题设定与入口链接，不转载外部营销页或课程大纲。

这一页的用处是从清单跳到可核对的官方入口；机制细节、API 与版本以官方文档为准。

## 工程实践

| 字段 | 内容 |
|------|------|
| 官方入口 | <https://rail-berkeley.github.io/bridgedata/> |
| 代码/仓库 | <https://github.com/rail-berkeley/bridge_data_v2> |
| 开源核查 | 以项目页 / GitHub 实际链接为准（清单可能滞后） |
| 源码运行时序图 | **不适用**（非论文可运行训练仓，或未核 README 入口） |
| 数据模态 | **多视角 RGB（部分含深度）+ 自然语言指令 + 末端执行器动作**（WidowX 机械臂真机遥操作）（据官方简介，以官方文档 / 数据卡为准） |
| 重定向就绪度 | **需适配**：真机操作演示自带动作标签，同构机型可直接训练；换本体须核对动作空间（末端位姿 vs 关节）与夹爪定义后再重定向（据清单与官方简介判断，以官方文档 / 数据卡为准） |

使用前先确认链接指向的是官方仓 / 文档，而不是镜像或过期 fork。

## 局限与风险

- 不要把 Awesome 摘要当成完整方法证明或合规结论。
- 同名 GitHub 仓（natnew vs aichr）条目链接可能不同；以本页主链接与技术地图为准。
- 清单中的实验室 / 硬件 / 人物条目偶发链到错误 org，复现或引用前先打开官方页核对。

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

- [vla](../methods/vla.md)
- [awesome-world-action-models-rcl](paper-rcl-wam-robot-learning-control-survey.md)
- [locomotion](../tasks/locomotion.md)

## 参考来源

- [`sources/repos/pai_awesome_dataset_085_bridgedata-v2.md`](../../sources/repos/pai_awesome_dataset_085_bridgedata-v2.md) — 本条目策展摘录
- [`sources/repos/awesome-physical-ai-union-catalog.md`](../../sources/repos/awesome-physical-ai-union-catalog.md) — 双清单并集目录
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)
- 主链接：<https://rail-berkeley.github.io/bridgedata/>

- [`sources/papers/rcl_awesome_wam_ref_d0a7d699e0efc759ae64_bridgedata-v2-a-dataset-for-robot-learni.md`](../../sources/papers/rcl_awesome_wam_ref_d0a7d699e0efc759ae64_bridgedata-v2-a-dataset-for-robot-learni.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://proceedings.mlr.press/v229/walke23a.html>

## 推荐继续阅读

- [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai)
- [aichr/awesome-physical-ai](https://github.com/aichr/awesome-physical-ai)
- [原文 / 官方入口](https://rail-berkeley.github.io/bridgedata/)

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://proceedings.mlr.press/v229/walke23a.html)
