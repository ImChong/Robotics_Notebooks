---
type: entity
tags: [agibot, simulation, benchmark, sim2real, open-source]
status: complete
updated: 2026-06-26
related:
  - ../overview/agibot-june-2026-release-technology-map.md
  - ../overview/agibot-release-category-02-sim-training-eval.md
  - ./go-2.md
  - ./paper-notebook-genie-sim-3-0-a-high-fidelity-comprehensive-simu.md
  - ../queries/simulator-selection-guide.md
  - ../overview/robot-training-stack-layers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
  - ../../sources/papers/humanoid_pnb_genie-sim-3-0-a-high-fidelity-comprehensive-simu.md
summary: "Genie Sim 3.0 是智元开源的仿真训练与评测平台：支持自然语言/图像生成可交互三维场景，提供 Genie Sim Benchmark 五类能力评测，并对接 RLinf、并行仿真与在线微调。"
---

# Genie Sim 3.0

**Genie Sim 3.0** 是智元 [AgibotTech/genie_sim](https://github.com/AgibotTech/genie_sim) 开源的 **高保真综合仿真平台**（入口：<https://agibot-world.com/genie-sim>）。在 [智元 2026-06 发布地图](../overview/agibot-june-2026-release-technology-map.md) 中承担 **仿真训练与评测** 段：把 **场景生成** 接到训练、评测与后训练，以降低真机试错成本。

## 一句话定义

**语言/图像驱动的三维场景生成 + 机器人能力 Benchmark + RL 训练接口**，目标是把仿真价值落到 **减少真机试错** 而非仅做可视化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| Gym | OpenAI Gym API | 强化学习标准环境接口惯例 |

## 为什么重要

- **场景生成：** 一句话或一张图生成可漫游、可训练的三维世界，并输出 RGB、深度、激光雷达等多模态数据。
- **Genie Sim Benchmark：** 五类核心能力——指令跟随、空间理解、操作执行、扰动适应、Sim2Real 迁移。
- **训练栈：** 支持 RLinf、并行仿真、标准 Gym 接口与 **在线微调**。
- **Sim2Real 叙事：** 材料称相同模型在仿真与真机评测差异 **<10%**（策展文建议等待更多任务验证）。

## 与 GE-Sim 2.0 / GO-2

| 项目 | 分工 |
|------|------|
| GE-Sim 2.0 | **视频世界模型** 闭环 |
| Genie Sim 3.0 | **场景级仿真** 训练与 benchmark |
| GO-2 | 文内在 Genie Sim Benchmark 与 Sim2Real 评测中报告领先结果 |

## 关联页面

- [仿真训练与评测分类 hub](../overview/agibot-release-category-02-sim-training-eval.md)
- [Paper Notebooks 索引实体](./paper-notebook-genie-sim-3-0-a-high-fidelity-comprehensive-simu.md)
- [训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)
- [humanoid_pnb_genie-sim-3-0-a-high-fidelity-comprehensive-simu.md](../../sources/papers/humanoid_pnb_genie-sim-3-0-a-high-fidelity-comprehensive-simu.md)

## 推荐继续阅读

- [Genie Sim GitHub](https://github.com/AgibotTech/genie_sim)
