---
type: entity
project_id: paper-agentic-teacher-2611-17394
tags: [paper,vla,robot-data,imitation-learning]
status: complete
updated: 2026-11-17
arxiv: "2611.17394"
original:
  - ../../sources/papers/agentic-teacher-2611-17394.md
sources:
  - ../../sources/papers/agentic-teacher-2611-17394.md
related:
  - ./paper-pear-robot-use-rl.md
  - ./paper-touchtwin-2611-17619.md
  - ./xhand-1-pro-u.md
summary: "从基础 VLA 的失败轨迹出发，使用编码智能体构建纠错控制器，再筛选可供基础策略学习的补救演示以进行后训练。"
---

# Agentic-Teacher: An Agentic Data Engine for Failure-Driven VLA Post-Training｜Agentic-Teacher：失败驱动的 VLA 后训练数据引擎

## 一句话定义

**Agentic-Teacher：失败驱动的 VLA 后训练数据引擎**：从基础 VLA 的失败轨迹出发，使用编码智能体构建纠错控制器，再筛选可供基础策略学习的补救演示以进行后训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作策略 |
| BC | Behavior Cloning | 模仿学习的行为克隆目标 |
| FM | Flow Matching | 动作生成模型的流匹配训练损失 |

## 为什么重要

常规数据扩容往往不针对当前策略的薄弱状态；失败驱动数据引擎把真实失败模式映射为纠错样本，使后训练更聚焦。

## 核心信息

| 项目 | 核查来源 |
|---|---|
| 作者或机构 | 原文作者机构，见 arXiv v1 |
| 原文 | [一手全文/原始新闻稿](https://arxiv.org/html/2611.17394v1) |
| 代码状态 | 未核实与本研究/设备对应的可运行代码仓库 |
| 核查版本 | arXiv:2611.17394v1 |

## 核心原理

分析失败 rollout，识别关键帧与失败类别；编码智能体编写可根据执行反馈修正动作的闭环程序；从失败前缀生成可恢复的成功轨迹。对轨迹加入时序相关扰动、去掉空转片段并保留夹爪切换，再使用冻结的基础 VLA 流匹配损失过滤不适配样本，微调后的 VLA 在部署时无需运行编码教师。

## 流程总览

```mermaid
flowchart LR
  A[原 VLA 失败 rollout]
  B[关键帧定位与失败分簇]
  C[编码智能体探索闭环纠错程序]
  D[成功续接轨迹]
  E[动作处理与流匹配损失筛选]
  F[后训练 VLA]
  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
```

上述流程图是基于原始资料整理的方法主干，不代表已复现作者模型。

## 源码运行时序图

**不适用：** 在本次核对的一手论文中未确认可运行的官方实现仓库及训练命令，不能把方法图改写为源码运行时序。

## 工程实践

1. 记录失败轨迹、关键状态和类别，验证纠错程序是否真的从失败状态完成任务。
2. 保留时间相关动作扰动与数据质量过滤；逐项比较去空帧和 flow-matching 过滤的贡献。
3. 固定基础 VLA、训练预算与测试任务，另记教师探索花费及可获得的纠错轨迹数量。

## 实验与评测

作者报告 π0 在 LIBERO-10 上由 79.4% 提升至 87.8%，在 MetaWorld-10 上由 80.4% 提升至 90.6%。均为原文特定仿真协议下的成功率。

## 与其他工作对比

相比 [PEAR](./paper-pear-robot-use-rl.md) 以交互奖励直接进行 RL，Agentic-Teacher 以教师程序产生纠错示范供 VLA 后训练，二者监督信号与部署开销不同。

## 结论

**从基础 VLA 的失败轨迹出发，使用编码智能体构建纠错控制器，再筛选可供基础策略学习的补救演示以进行后训练。**

1. 先根据失败原因确定需要新增哪些样本，而不是盲目扩充全部训练集。
2. 程序控制器完成任务不代表原 VLA 能模仿成功，过滤与扰动处理不可略过。
3. 分开统计训练时生成数据成本与推理时最终 VLA 开销。

## 局限与风险

纠错质量依赖编码教师覆盖、仿真物理一致性和冻结策略筛选器；论文结果不能直接推出未见真实机器人的成功率。

## 关联页面

- [paper pear 2611 18562](./paper-pear-robot-use-rl.md)
- [paper touchtwin 2611 17619](./paper-touchtwin-2611-17619.md)
- [xhand 1 pro u](./xhand-1-pro-u.md)
- [模仿学习](../methods/imitation-learning.md)
- [数据增广](../methods/generative-data-augmentation.md)
- [vla 路线](../../roadmap/depth-vla.md)

## 参考来源

- [一手原文核对记录](../../sources/papers/agentic-teacher-2611-17394.md)

## 推荐继续阅读

- [一手正文](https://arxiv.org/html/2611.17394v1)
- [作者/扩展资料](https://arxiv.org/abs/2611.17394)

