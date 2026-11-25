---
type: overview
tags: [agibot, simulation, benchmark, sim2real, category-hub, survey]
status: complete
updated: 2026-11-21
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
original:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
---

# 智元发布分类 02：仿真训练与评测

> **分类页**：**02 仿真训练与评测**；总地图见 [智元 2026-06 发布技术地图](./agibot-june-2026-release-technology-map.md)。

## 一句话定义

**本分类汇总智元 2026-06 发布中面向仿真训练、自动化评测与真机前闭环微调的平台层能力，核心代表为 Genie Sim 3.0。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| Gym | OpenAI Gym API | 强化学习标准环境接口惯例 |

## 为什么重要

**真机试错太贵时，仿真能否承担训练、评测与后训练？** Genie Sim 3.0 把「自然语言/图片生成三维可交互场景 → 标准化 Benchmark 评测 → 分布式 RLinf 并行训练与在线微调」串成同一条底座链路，直接服务 VLA 与通用操作策略的规模化迭代。

## 核心原理与本组项目

| # | 项目 | Wiki 实体 |
|---|------|-----------|
| 02 | Genie Sim 3.0 | [genie-sim-3.md](../entities/genie-sim-3.md) |

### 文内强调能力

| 能力 | 含义 |
|------|------|
| 场景生成 | 自然语言或图片 → 可交互三维世界 |
| Genie Sim Benchmark | 指令跟随、空间理解、操作执行、扰动适应、Sim2Real 五类 |
| 训练接口 | RLinf、并行仿真、Gym 接口、在线微调 |
| Sim2Real 指标 | 公开介绍称仿真与真机评测成功率差异 **<10%**（需更多任务与硬件独立验证） |

## 工程实践

- **以 Gym 接口解耦训练器与场景资产：** 通过标准化观测/动作 Gym 接口对接 RLinf 等强化学习或后训练框架，避免场景生成器与策略训练脚本紧耦合。
- **五维度回归防单指标过拟合：** 同时跟踪指令跟随、空间理解、操作执行、扰动适应与 Sim2Real 一致性，避免只在固定光照或单物品摆放上刷高分数。
- **在线微调前先做小样本实机校准：** 即使宣传称仿真与真机差异 <10%，跨物理接触任务仍需保留少量真机 rollout 对齐摩擦、阻尼与相机延迟。

## 常见误区或局限

- **把公关口吻当独立基准：** 「仿真与真机差异 <10%」源自发布材料统计，尚未提供独立第三方跨本体全量消融细节。
- **只看仿真吞吐忽略长尾接触保真：** 生成式三维场景若未精细校准接触参数与材质物理属性，长程精细装配仍会出现 Sim2Real 落差。

## 关联页面

- [GO-2](../entities/go-2.md) — 文内在 Genie Sim Benchmark 与 Sim2Real 评测中对比
- [训练栈分层技术地图](./robot-training-stack-layers-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)

## 推荐继续阅读

- [Genie Sim GitHub](https://github.com/AgibotTech/genie_sim)
