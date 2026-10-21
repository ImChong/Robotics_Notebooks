---
type: entity
project_id: dreamcan-sparse-visual-planning
tags: [paper, robot-manipulation, long-horizon-planning, video-model]
status: complete
updated: 2026-10-21
arxiv: "2610.24981"
project: "https://dreamcan.github.io"
related:
  - ../entities/paper-hint-robot-manipulation.md
  - ../entities/unifolm-world-model-action.md
sources:
  - ../../sources/papers/dreamcan_arxiv_2610_24981.md
  - ../../sources/sites/dreamcan.md
summary: "DreamCan 将预训练视频模型用作长程操作高层规划器，生成稀疏视觉子目标，再由低层视觉运动策略执行；论文报告 LIBERO-Long 成功率为 91.2%。"
---

# DreamCan：长程操作的稀疏视觉规划

## 一句话定义

DreamCan 从预训练视频模型产生稀疏视觉子目标，由低层 visuomotor 策略执行，以把长程操作规划拆成可达的小阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DreamCan | Dream What I Can | 以稀疏视觉规划连接视频模型与机器人执行的分层策略 |
| LIBERO | Lifelong Benchmark for Robot Manipulation | 论文用于评测多任务与长程操作的仿真基准 |
| DP | Dynamic Programming | 从候选子目标中优化选择序列的动态规划 |
| VMP | Visuomotor Policy | 根据视觉输入生成机器人动作的低层策略 |

## 为什么重要

视频模型可以预测看似合理的未来画面，但视觉上合理的轨迹未必可由机器人执行。DreamCan 通过少量、可达的视觉目标约束高层规划，并由专门的低层策略完成动作落地。

## 核心原理

从无动作标签的机器人视频估计子目标的可达性和任务效用，再用动态规划选取有用的子目标序列。高层输出稀疏视觉状态而非每一帧动作，低层策略把当前观测与目标转换成具体运动。

## 评测与结果

作者报告 Language-Table 任务的规划结果，以及 LIBERO-Long 上 **91.2%** 成功率；低层策略为 0.1B 参数。论文还展示未见物体组合下的真实双臂折盒、装箱任务。该数字应连同数据、任务和成功判据一起引用。

## 与其他工作对比

论文比较语言规划器、稠密视频规划和均匀间隔子目标规划。其主要区别是按可达性和任务效用选择子目标，而不是固定密度生成未来帧。论文摘要报告整体优于这些对照，但完整复现仍需核对实验设置。

## 工程实践与开源状态

| 项目 | 现状 |
|------|------|
| 规划器 | 预训练视频模型适配为高层稀疏子目标生成器 |
| 子目标 | 依据可达性与任务效用动态规划选择 |
| 执行器 | 0.1B 参数视觉运动低层策略 |
| 代码 | arXiv 页面未列代码仓库；项目页可达性与源码入口待复核 |

### 源码运行时序图

**不适用：** 当前没有核验到官方代码仓库或可运行入口，待项目页可访问后补充。

## 局限与风险

- 高层目标视觉可达并不保证低层策略在新场景中能可靠到达。
- 动态规划的可达性和任务效用估计质量会影响子目标链。
- 91.2% 是特定 LIBERO-Long 设置的结果，不应外推为一般真实机器人成功率。
- 项目页代码状态仍待核查，本文档依据 arXiv 摘要整理。

## 结论

**DreamCan 的关键取舍是用更少但更可达的视觉子目标，替代难执行的稠密视频轨迹规划。**

- 优先关注子目标可达性与任务效用如何估计，这是高层规划的核心。
- 91.2% 仅对应论文的 LIBERO-Long 评测口径。
- 真实双臂任务展示了应用潜力，但摘要未给出足够细节来评估跨任务泛化范围。
- 未核实源码前，不应将论文指标视作开箱可复现性能。

## 关联页面

- [HINT 长程操作](../entities/paper-hint-robot-manipulation.md) — 通过语义子任务和跟踪维持长程意图
- [UnifoLM 世界模型动作](../entities/unifolm-world-model-action.md) — 世界模型与动作策略的结合路线

## 参考来源

- [DreamCan 论文摘录](../../sources/papers/dreamcan_arxiv_2610_24981.md)
- [DreamCan 项目页归档](../../sources/sites/dreamcan.md)

## 推荐继续阅读

- [DreamCan arXiv 页面](https://arxiv.org/abs/2610.24981)
- [DreamCan 项目页](https://dreamcan.github.io)
