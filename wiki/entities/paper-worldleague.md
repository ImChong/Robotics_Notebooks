---
type: entity
project_id: worldleague
arxiv: "2611.30989"
project: https://xyangzhou.github.io/WorldLeague/
code: https://github.com/xyangzhou/WorldLeague
tags: [benchmark, embodied-ai, vla, nus]
status: draft
updated: 2026-11-25
summary: "通过轨迹上的物理目标条件，在共享任务上比较使用不同动作接口的 VLM、VLA 与 WAM。"
related:
  - ./embodied-first-exam.md
  - ./paper-embo-harness.md
  - ./paper-uni-robot-agent.md
  - ../../roadmap/depth-embodied-eval.md
sources:
  - ../../sources/papers/worldleague.md
  - ../../sources/sites/worldleague.md
  - ../../sources/repos/worldleague.md
original:
  - ../../sources/papers/worldleague.md
---

# Benchmarking Physical Agency (WorldLeague)｜具身系统物理目标达成能力评测

## 一句话定义

通过轨迹上的物理目标条件，在共享任务上比较使用不同动作接口的 VLM、VLA 与 WAM。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLM | Vision-Language Model | 视觉语言模型 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| WAM | World Action Model | 世界动作模型 |
| QA | Question Answering | 问答 |

## 核心信息

| 项目 | 内容 |
|---|---|
| 机构 | 新加坡国立大学 xML Lab |
| 发布 | 2026-11-23（v1） |
| 原文 | [arXiv:2611.30989](https://arxiv.org/abs/2611.30989) |
| 项目 | [主页](https://xyangzhou.github.io/WorldLeague/) |
| 源码 | [GitHub](https://github.com/xyangzhou/WorldLeague) |

## 核心原理

60 项任务分为 WorldSeek、WorldShape、WorldShift，覆盖导航、定位、几何、物理、记忆、控制。VLM 经 WorldHarness 选择动作，连续策略使用自身接口；从物理轨迹计算目标项完成度及全部条件满足率，终端 QA 单列，不代替物理分数。

## 流程总览

```mermaid
flowchart TD
  A["物理任务与场景"] --> B["VLM：WorldHarness"]
  A --> C["VLA / WAM：原生接口"]
  B --> D["物理轨迹"]
  C --> D
  D --> E["目标项与约束判定"]
  E --> F["Interactive Score 与成功率"]
  B --> G["终端 QA 单独评分"]
```

## 源码运行时序图

```mermaid
sequenceDiagram
  participant U as worldleague CLI
  participant C as worldleague.cli
  participant P as continuous.plan
  participant R as run_continuous_episode.main
  participant S as score.profile
  U->>C: continuous model task
  C->>P: plan
  P-->>C: 运行计划
  C->>C: 写入 plan.json
  C->>R: main --plan
  U->>C: profile outputs
  C->>S: result.json 记录集合
```

`worldleague` 指向 `worldleague.cli:main`；已读 `src/worldleague/cli.py`：eval 可进入 `evaluation.r1_session.evaluate`，continuous 生成 plan.json 后调用 `runtime.commands.run_continuous_episode.main`；profile 从 result.json 聚合；未执行。 此图仅覆盖已核对入口，内部完整调用关系待复核。

## 评测与实验

13 个系统中最高平均 Interactive Score 28.58/100，完成 13/60；347 个终端 QA 全对的 VLM 回合中 249 个物理分数为零（71.8%）。连续控制评估覆盖 59 项，跨系统比较要同时注明接口、任务覆盖和预算。

## 结论

当前证据支持将此工作作为相应方法的工程参考，而非通用真机性能保证。

- 物理成功须同时满足任务目标与约束。
- 保留部分进度与终端 QA 独立指标。
- 报告各系统接口、覆盖任务和资源预算。

## 局限与风险

仅仿真，不代表真机成功率。推理期间物理环境继续运行，推理时间和接口抽象影响比较。官方代码和基准数据已列出；场景生成公开入口在 README 中仍为 TODO。

> 草稿：原文与必要源码未全部通读，尚未运行仓库健康度脚本或 preflight。

## 关联页面

- [embodied-first-exam](./embodied-first-exam.md)
- [paper-embo-harness](./paper-embo-harness.md)
- [paper-uni-robot-agent](./paper-uni-robot-agent.md)
- [纵深路线](../../roadmap/depth-embodied-eval.md)

## 推荐继续阅读

- [官方项目](https://xyangzhou.github.io/WorldLeague/)

## 参考来源

- [论文归档](../../sources/papers/worldleague.md)
- [项目页归档](../../sources/sites/worldleague.md)
- [源码归档](../../sources/repos/worldleague.md)
