---
type: entity
project_id: coda-coevolving-data-policy
tags: [paper, robotics, data-flywheel, rl, ingest-2026-11]
status: complete
updated: 2026-11-15
topic: [teleop-data, reinforcement-learning]
arxiv: "2610.22362"
project: https://plan-lab.github.io/projects/coda/
related:
  - ../concepts/embodied-data-flywheel-minimal-closed-loop.md
  - ../concepts/data-flywheel.md
  - ../../roadmap/depth-embodied-data.md
  - ./paper-prm4rl-progress-reward.md
  - ./paper-demostream-batched-tamp.md
original:
  - ../../sources/papers/coda-coevolving-data-policy_arxiv_2610_22362.md
sources:
  - ../../sources/papers/coda-coevolving-data-policy_arxiv_2610_22362.md
  - ../../sources/sites/coda-coevolving-data-policy.md
summary: "根据失败诊断联合更新仿真数据与策略"
---

# CoDA：失败驱动的数据引擎与动作策略协同进化

## 一句话定义

**通过共享的失败诊断和学习进展记忆，同时生成下一轮训练场景与势函数监督，使仿真数据引擎和机器人策略共同改进。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CoDA | Co-Evolution of Data Engines and Action Policies | 数据和策略共同进化 |
| OOD | Out-of-Distribution | 分布外泛化 |
| SFT | Supervised Fine-Tuning | 监督微调 |

## 为什么重要

根据失败诊断联合更新仿真数据与策略，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；已通读原始论文 HTML 的可用正文和附录；源码执行效果仍需复现。

## 核心机制

记录 rollout 的阶段化失败与场景有效性，诊断为 difficulty mode；同一诊断编译成新的场景生成程序和 Text2potential 奖励/价值学习信号，并按学习增益调度数据生成预算。

## 方法流程

该流程依据原文方法描述，不等价于源码调用时序。

```mermaid
flowchart LR
  rollout[策略执行与失败轨迹] --> critic[诊断失败模式]
  critic --> memory[共享异质记忆]
  memory --> scenes[难度针对性场景生成]
  memory --> reward[Text2potential 势函数监督]
  scenes --> train[策略更新]
  reward --> train
  train --> rollout
```

## 源码运行时序图

**暂不适用**：未核实到官方已发布的完整可运行实现，不以通用模板替代实际调用链。

## 实验与评测

四类操纵任务 OOD 成功率 27.6%，对应 SFT 基线 5.3%、按任务成功率调度的数据引擎 18.1%；LIBERO-Pro 的位置扰动结果 25%→84%、指令扰动 1%→76%。主要为仿真结果，不能外推为真机效果。

## 局限与工程实践

- 依赖可用仿真环境、可验证失败诊断及场景生成程序；项目页标注 Code coming soon，完整复现暂受限。
- **源码状态：** 官方项目页 2026-11-14 检查：Code (coming soon)，尚未提供可用源码地址。
- **原文状态：** 已通读 arXiv HTML 的可用正文和附录；具体复现及跑通情况尚未独立验证。

## 结论

**失败日志只有进入下一轮场景生成和策略训练的共同闭环，才有机会转化为泛化收益。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 与相关方法对比

| 对照方法 | 关键区别 | 局限或适用条件 |
|---|---|---|
| 只按成功率调度采数 | 单一成败信号不足以解释失败模式 | CoDA 诊断失败并生成下一轮场景与训练势函数 |
| 数据引擎和策略分开更新 | 两方使用的反馈目标可能不一致 | CoDA 用共享记忆联动更新 |

不同基准和硬件的结果不能直接排序。

## 关联页面

- [具身数据飞轮最小闭环](../concepts/embodied-data-flywheel-minimal-closed-loop.md)：策略失败进入下一轮数据生成与训练是 CoDA 的核心闭环。
- [Data Flywheel（具身数据飞轮）](../concepts/data-flywheel.md)：对应数据生成、策略更新、评估反馈的循环。
- [具身数据纵深路线](../../roadmap/depth-embodied-data.md)：理解失败驱动仿真生成在数据技术路线中的位置。
- [PRM4RL（进度奖励）](./paper-prm4rl-progress-reward.md)：都使用势函数信号，但 CoDA 还针对失败模式改变训练场景。
- [DemoStream（规划示范生成）](./paper-demostream-batched-tamp.md)：比较失败驱动场景生成与批量规划示范两种数据来源。
- [Robot Simulation Scene Generation · 机器人仿真场景生成](../methods/robot-simulation-scene-generation.md)：CoDA 基于策略 rollout 的失败模式诊断动态生成针对性仿真训练场景，将数据引擎与策略学习纳入同一闭环。
- [机器人策略学习流程 · Robot Policy Learning](../methods/robot-policy-learning.md)：该框架同步利用失败记忆生成场景与势函数监督信号驱动 SFT 和 RL 训练，提升机器人操作策略的分布外泛化能力。

## 参考来源

- [论文归档](../../sources/papers/coda-coevolving-data-policy_arxiv_2610_22362.md)
- [官方项目页归档](../../sources/sites/coda-coevolving-data-policy.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2610.22362)
- [官方项目页](https://plan-lab.github.io/projects/coda/)
