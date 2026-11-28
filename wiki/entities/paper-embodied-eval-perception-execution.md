---
type: entity
project_id: embodied-intelligence-model-evaluation-jig-250550
tags: [paper, embodied-ai, evaluation, benchmark, real-robot]
status: complete
updated: 2026-11-28
doi: "10.11834/jig.250550"
venue: "中国图象图形学报 31(8), 2026"
summary: "静态、仿真和真机分层评测；Octo/UR5 对照量化静态关键帧与动态表现的相关性，但不支持以静态代替真机。"
original:
  - ../../sources/papers/embodied_eval_perception_execution_jig_250550.md
sources:
  - ../../sources/papers/embodied_eval_perception_execution_jig_250550.md
related:
  - ../methods/embodied-agent-benchmarking.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ./paper-simplerenv-real2sim-eval.md
---

# 具身智能模型评测：从感知到执行 / Embodied Intelligence Model Evaluation: From Perception to Execution

## 一句话定义

综述用三级评测区分静态筛查、仿真闭环与真机验证，并用 Octo/UR5 对照检验静态关键帧能代理多少实际动作表现。

## 英文缩写速查

| 缩写 | 英文全称 | 含义 |
|---|---|---|
| VLA | Vision–Language–Action | 结合视觉语言输入并输出机器人动作 |
| PLCC | Pearson Linear Correlation Coefficient | 静态与真机评分的线性相关 |
| S2D | Static-to-Dynamic | 本文定义的静态到动态一致性 |

## 为什么重要

同样的排名不代表每一次都做对。静态快测可以提前筛掉部分错误，却看不到动态接触和累积误差；阶段之间应有可以量化的桥梁。

## 核心原理与流程

五类基础任务（拾取、放置、按压、拉、推）共 50 个实例。真机每实例做 50 轮，成功、达到 50 步或安全停止即结束；静态侧用 Octo 对关键帧输出直接评分。先对动作维度分别计算相关，再给整体汇总。

```mermaid
flowchart LR
    A["五类任务，共 50 个实例"] --> B["Octo 静态关键帧评分"]
    A --> C["Octo 在 UR5 上闭环执行"]
    C --> D["按终止后的实际状态评分"]
    B --> E["分别比较位置、旋转与夹爪"]
    D --> E
    E --> F["Pearson r 分项与均值"]
```

## 三级评测各回答什么

| 范式 | 最适合核验 | 局限 |
|---|---|---|
| 静态数据集 | 多模态感知、语义理解和关键帧动作预测 | 不能观察动作后的状态反馈、误差累积或恢复 |
| 交互仿真 | 可控变量、条件扰动、规划与闭环策略，便于高并行复测 | 传感、接触与控制模型不一定代表部署环境 |
| 真实机器人 | 操作精度、物理接触、能耗和人机交互 | 条件复现、平台差异和采集成本限制覆盖率 |

感知关注多模态输入保真与对齐；认知考查语义、对象关系与分布外表征；决策看规划、因果与安全约束；执行记录精度、接触或碰撞及累积误差。跨标准时应写清任务定义、硬件、成功判据、错误来源和成本，不建议把三级数值直接相加。

## 实验与局限

图 2 位置、旋转、夹爪 r 分别为 **0.629、0.621、0.729**，汇总约 **0.66**。表 6 的放置位置/旋转为 **0.471/0.343**，而按压旋转是 **0.825**，不能将总均值当成每种任务的一致性。§6 只给静态和真机的直接对照，没有给仿真的第三组数字。单一模型和平台、基础任务及线性相关都限制了外推。

## 与其他工作对比

- [SIMPLER](paper-simplerenv-real2sim-eval.md) 比较闭环仿真与真机的策略相对表现；本研究比较静态关键帧与真机的动作维度表现，两个 r 的样本单位和系统条件不同。
- [RoboChallenge Zero](paper-robochallenge-zero.md) 直接测新任务完整系统；该综述的对照适合先筛诊断样本，而不是替代真机试验。

## 结论

先用便宜评测筛查，再用可控仿真定位原因，最后用真实装置验收。

1. 静态分项按位置、旋转、夹爪拆开。
2. 将配对任务、停止条件和计分规则写入报告。
3. 用真机补查放置等低一致性子类，不以 0.66 充当成功率。

## 工程实践

期刊页面未提供 §6 独立可运行的评测实现；上图仅是实验设计，不是源码时序。

## 关联页面

- [具身智能体评测](../methods/embodied-agent-benchmarking.md)
- [仿真与真机评测差距](../concepts/sim-vs-real-eval-gap.md)

## 参考来源

- [期刊原文核对归档](../../sources/papers/embodied_eval_perception_execution_jig_250550.md)

## 推荐继续阅读

- [具身评测纵深路线](../../roadmap/depth-embodied-eval.md)
