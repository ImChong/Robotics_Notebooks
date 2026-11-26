---
type: entity
tags: [curated-index, physical-ai, awesome-physical-ai, eval]
status: complete
updated: 2026-09-20
code: https://github.com/motional/nuplan-devkit
summary: "End-to-end planning evaluation stack with documented metrics and simulation loops."
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ../methods/vla.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/pai_awesome_eval_112_nuplan-devkit.md
  - ../../sources/repos/awesome-physical-ai-union-catalog.md
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
---

# nuPlan Devkit

**nuPlan Devkit** 收录于 awesome-physical-ai（natnew）**第 112/384** 条，分组 **Evaluation Methodology**。本页是 **清单索引**：给出它在清单中的位置与官方入口，细节以官方文档 / 原文为准。

## 一句话定义

End-to-end planning evaluation stack with documented metrics and simulation loops.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 具身/物理智能策展主题 |
| RL | Reinforcement Learning | 评测统计口径常用于 RL |
| VLA | Vision-Language-Action | 通才策略评测对象 |

## 为什么重要

- End-to-end planning evaluation stack with documented metrics and simulation loops.
- 想横向对照同一分组的其他工作，可以从 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md) 逐条展开。
- 两份清单合并去重：同一条目只有这一页，出处见下方「参考来源」。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 112/384 |
| 分组 | Evaluation Methodology |
| 来源清单 | natnew |
| 主链接 | <https://github.com/motional/nuplan-devkit> |
| 代码/仓库 | <https://github.com/motional/nuplan-devkit> |

## 核心原理

End-to-end planning evaluation stack with documented metrics and simulation loops.

该条目在 Physical AI 清单中的角色是 **eval**，分组 **Evaluation Methodology**。本页只给出清单里的问题设定与入口链接，不转载外部营销页或课程大纲。

这一页的用处是从清单跳到可核对的官方入口；机制细节、API 与版本以官方文档为准。

## 工程实践

| 字段 | 内容 |
|------|------|
| 官方入口 | <https://github.com/motional/nuplan-devkit> |
| 代码/仓库 | <https://github.com/motional/nuplan-devkit> |
| 开源核查 | 以项目页 / GitHub 实际链接为准（清单可能滞后） |
| 源码运行时序图 | **不适用**（非论文可运行训练仓，或未核 README 入口） |

使用前先确认链接指向的是官方仓 / 文档，而不是镜像或过期 fork。

## 机制与架构图

```mermaid
flowchart LR
    A["端到端规划策略"] --> B["nuPlan Devkit 规划评测栈 (natnew 112/384)"]
    B --> C["官方仓库 (github.com/motional/nuplan-devkit)"]
    B --> D["仿真循环执行与文档化指标评测"]
    D --> E["关联方向: RL / VLA 与 Sim2Real"]
```

## 局限与风险

- 不要把 Awesome 摘要当成完整方法证明或合规结论。
- 同名 GitHub 仓（natnew vs aichr）条目链接可能不同；以本页主链接与技术地图为准。
- 清单中的实验室 / 硬件 / 人物条目偶发链到错误 org，复现或引用前先打开官方页核对。

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)

- [AI 模型与智能体基准评测](../methods/ai-benchmark-evaluation.md)：了解测试任务、协议与指标。
- [轨迹优化](../methods/trajectory-optimization.md)：了解运动约束与轨迹求解。

## 参考来源

- [`sources/repos/pai_awesome_eval_112_nuplan-devkit.md`](../../sources/repos/pai_awesome_eval_112_nuplan-devkit.md) — 本条目策展摘录
- [`sources/repos/awesome-physical-ai-union-catalog.md`](../../sources/repos/awesome-physical-ai-union-catalog.md) — 双清单并集目录
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)
- 主链接：<https://github.com/motional/nuplan-devkit>

## 推荐继续阅读

- [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai)
- [aichr/awesome-physical-ai](https://github.com/aichr/awesome-physical-ai)
- [原文 / 官方入口](https://github.com/motional/nuplan-devkit)
