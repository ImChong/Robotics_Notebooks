---
type: overview
tags: [agibot, simulation, benchmark, sim2real, category-hub, survey]
status: complete
updated: 2026-06-26
summary: "智元 2026-06 发布 · 02 仿真训练与评测 — Genie Sim 3.0 如何把场景生成接到训练、评测与在线微调？"
related:
  - ./agibot-june-2026-release-technology-map.md
  - ./agibot-release-category-01-data-entry.md
  - ./agibot-release-category-03-world-model.md
  - ../entities/genie-sim-3.md
  - ../entities/go-2.md
  - ../queries/simulator-selection-guide.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
---

# 智元发布分类 02：仿真训练与评测

> **图谱分类节点**：**02 仿真训练与评测**；总地图见 [智元 2026-06 发布技术地图](./agibot-june-2026-release-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| Gym | OpenAI Gym API | 强化学习标准环境接口惯例 |

## 核心问题

**真机试错太贵时，仿真能否承担训练、评测与后训练？** Genie Sim 3.0 强调 **语言/图像生三维场景** 并接入 **Benchmark + RLinf + 在线微调**。

## 本组项目（1 个）

| # | 项目 | Wiki 实体 |
|---|------|-----------|
| 02 | Genie Sim 3.0 | [genie-sim-3.md](../entities/genie-sim-3.md) |

## 文内强调能力

| 能力 | 含义 |
|------|------|
| 场景生成 | 自然语言或图片 → 可交互三维世界 |
| Genie Sim Benchmark | 指令跟随、空间理解、操作执行、扰动适应、Sim2Real 五类 |
| 训练接口 | RLinf、并行仿真、Gym 接口、在线微调 |
| Sim2Real 指标 | 材料称仿真与真机评测差异 **<10%**（须更多任务验证） |

## 关联页面

- [GO-2](../entities/go-2.md) — 文内在 Genie Sim Benchmark 与 Sim2Real 评测中对比
- [训练栈分层技术地图](./robot-training-stack-layers-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)

## 推荐继续阅读

- [Genie Sim GitHub](https://github.com/AgibotTech/genie_sim)
