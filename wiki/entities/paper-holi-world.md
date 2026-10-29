---
type: entity
project_id: holi-world
tags: [paper, spatial-intelligence, perception]
status: complete
updated: 2026-10-27
arxiv: "2610.26549"
project: "https://ropedia.github.io/holi-world/"
code: "https://github.com/Ropedia/holi-world"
related:
  - ./paper-ropedia-w0.md
  - ./paper-donic.md
sources:
  - ../../sources/papers/holi-world.md
  - ../../sources/sites/holi-world.md
  - ../../sources/repos/holi-world.md
summary: "将第一视角记录转换为静态三维对象、动态轨迹与时空推理监督。"
---

# Holi-World：从第一视角经验构建世界理解

## 一句话定义

将第一视角记录转换为静态三维对象、动态轨迹与时空推理监督。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QA | Question Answering | 问答监督 |
| VL | Vision-Language | 视觉语言模型 |
| AP | Average Precision | 目标检测指标 |

## 核心原理

Foundry 结合静态场景感知和动态对象跟踪，生成关联几何、语义和交互历史的世界记录；Data 提供约 100 个场景和约 8 万条问答，VL-4B 学习时空推理，Bench 使用独立人工标注评估。

## 评测与结果

作者在 ScanNet 等数据集与自建基准评估静态检测、动态跟踪及推理；项目页报告 ScanNet AP50 0.55，需按对应评测协议解读。

## 结论

将第一视角记录转换为静态三维对象、动态轨迹与时空推理监督。

- 优先核对评测协议、输入条件与基线设置，再比较指标。
- 场景几何质量和感知误差会影响下游结果。
- 项目结果不能直接等同于真实机器人闭环任务表现。

## 源码运行时序图

不适用：官方仓库当前没有可运行的项目实现。

## 工程实践与局限

官方仓库 README 表明代码不计划开源；模型和数据列为待发布，仓库目前仅提供项目说明。

## 与其他工作对比

| 工作 | 场景表示 | 与 Holi-World 的差异 |
|------|----------|----------------------|
| [ConceptGraphs](./paper-conceptgraphs-open-vocabulary-3d-scene.md) | 开放词汇三维对象图 | ConceptGraphs 把 VLM 语义融入静态对象图供空间查询；Holi-World 在静态对象之外追加动态轨迹与交互历史，并据此生成时空问答监督训练 VL-4B |
| [SpatialLM](./cn-os-spatiallm.md) | 点云 → 结构化室内布局与 3D 框 | SpatialLM 输出静态布局且训练代码与合成数据公开；Holi-World 面向第一视角记录中的动态交互，代码不计划开源 |
| [S-Agent 2.0](./paper-s-agent-2.md) | 推理时调用感知工具 + 场景记忆 | 同一团队的另一条路线：S-Agent 2.0 不训练模型，靠工具读数推理；Holi-World 把世界记录蒸馏成训练数据，训练专用 VL 模型 |

ScanNet AP50 只衡量静态检测；若关心动态物体与时序推理，应看自建 Bench 的分项结果。在[感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md)中，它对应「2D→3D 提升与语义建图」层，并向下游提供带时间维度的语义地图。

## 关联页面

- [Ropedia W0](./paper-ropedia-w0.md) — 跨本体第一视角世界模型。
- [DONIC](./paper-donic.md) — 从人类示范学习双手跟踪控制。
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 感知栈选型闭环知识链枢纽

## 参考来源

- [论文归档](../../sources/papers/holi-world.md)
- [项目页归档](../../sources/sites/holi-world.md)
- [官方仓库归档](../../sources/repos/holi-world.md)

## 推荐继续阅读

- [论文原文](https://arxiv.org/abs/2610.26549)
- [官方仓库](https://github.com/Ropedia/holi-world)
