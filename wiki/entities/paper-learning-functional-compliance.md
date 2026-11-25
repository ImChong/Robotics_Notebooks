---
type: entity
project_id: learning-functional-compliance
openreview: https://openreview.net/forum?id=wALhOwQWnx
venue: CoRL 2026
tags: [paper, humanoid, compliance, loco-manipulation, rl]
status: draft
updated: 2026-11-25
related:
  - ../tasks/loco-manipulation.md
  - ../methods/hierarchical-reinforcement-learning.md
  - ./paper-ceer2-directional-compliance.md
sources:
  - ../../sources/papers/learning-functional-compliance-corl-2026.md
original:
  - ../../sources/papers/learning-functional-compliance-corl-2026.md
summary: "依据公开摘要，功能性柔顺按任务方向和时间调节控制响应，在保持任务所需控制能力的同时允许其他方向顺应。全文待核验。"
---

# Learning Functional Compliance in Whole-body Manipulation｜全身操作中的功能性柔顺学习

## 一句话定义

功能性柔顺（Functional Compliance）针对任务方向和执行阶段选择柔顺响应，而非所有方向一律放松控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| WBC | Whole-Body Control | 全身控制 |
| RL | Reinforcement Learning | 强化学习 |
| CoRL | Conference on Robot Learning | 机器人学习会议 |

## 为什么重要

接触操作不一定需要机器人在所有方向都同样硬或同样软。拿插入任务举例，推进需要维持控制能力，横向接触却可能需要让位；若从接触过渡到其他阶段，两者的相对重要性也可能改变。这是用来理解任务取舍的例子，不代表论文公布了具体轴向设置。摘要报告钥匙插入和书本重定向获益，而开柜门与最强基线相当，因此不宜把它解读为所有接触任务都需要时变柔顺。

## 核心信息

| 项目 | 内容 |
|---|---|
| 论文 | Learning Functional Compliance in Whole-body Manipulation |
| 作者 | Gio Huh, Yue Zhuo, Zehua Wang, Alan Yu, Yibo Cheng, Yanbing Han, Gabriel B. Margolis, Ge Yang |
| 机构 | FortyFive Labs、麻省理工学院、斯坦福大学、加州理工学院（用户提供，待论文首页核验） |
| 会议 | CoRL 2026 Poster |
| 平台 | Unitree G1 |
| 原文核对 | 未完成；OpenReview 访问验证 |

## 核心机制（仅摘要线索）

- 根据任务轴与阶段调节柔顺性，使必要的运动方向保持控制能力。
- 学习控制策略，并以柔顺位姿塑形奖励；同时学习力估计器，部署时用于组合可变柔顺配置。
- 具体策略网络、力估计器输入、控制接口、损失函数与训练过程尚未核对，不据摘要补写。

## 流程总览

待阅读全文后绘制基于方法章节的 Mermaid 流程图。本草稿不推定未核实的模块连线。

## 源码运行时序图

暂不适用：尚未确认官方可运行代码或运行入口；源码可用性待核实。

## 工程实践

以下是基于摘要的设计与评测提示，不是论文的部署步骤。

- **任务建模：**先标明各阶段需要维持运动的方向和允许顺应的方向；不要把一个全局刚度值当作唯一设计变量。
- **对照设计：**在同一任务、初始条件和成功判据下，对比统一柔顺与任务相关的可变柔顺。分开记录动作跟踪精度和完成率；跟踪精度相近并不说明接触阶段同样可靠。
- **复现前核对：**摘要只确认力估计器参与部署，不说明其所需信号、评估精度或怎样组合柔顺配置。获得原文后先核对这些接口及真机安全约束，再确定可实施的复现方案。

## 实验与评测（摘要报告）

| 任务 | 摘要报告的相对结果 |
|---|---|
| 钥匙插入 | 方向与时变柔顺带来成功率改善 |
| 书本重定向 | 成功率改善 |
| 开柜门 | 与最强基线相当 |
| 动作跟踪 | 精度与先进方法相当 |

尚未核实成功率数值、统计样本、具体基线与实验协议。

## 结论

摘要提示任务相关的可变柔顺是值得进一步研究的全身操作控制接口；当前结论仅为待核验线索。

- 阅读全文时分开记录任务方向与非任务方向的控制目标。
- 对比力估计误差与操作任务成功率，核对二者的评测条件。
- 核对时变柔顺对不同任务的收益、控制精度和真实部署边界。

## 局限与风险

OpenReview 论坛与 PDF 均触发访问验证，未读到方法、实验及局限章节；未确认代码、数据与权重开放状态。上述内容不能作为复现依据。

## 关联页面

- [移动操作](../tasks/loco-manipulation.md)
- [分层强化学习](../methods/hierarchical-reinforcement-learning.md)（相关方法背景，不代表本文采用分层架构）
- [CEER2：方向可调的人形末端与根部柔顺](./paper-ceer2-directional-compliance.md)（方向柔顺的相关工作，待全文对比）
- [接触丰富操作纵深路线](../../roadmap/depth-contact-manipulation.md)

## 参考来源

- [论文来源归档](../../sources/papers/learning-functional-compliance-corl-2026.md)
- [OpenReview](https://openreview.net/forum?id=wALhOwQWnx)
- [作者主页](https://zehuaw1.github.io/)
- [会议摘要索引（转载）](https://hongsong-wang.github.io/CoRL2026/)

## 推荐继续阅读

- [论文 PDF](https://openreview.net/pdf/aced2ffb6232ae3ddfe52e5e013b4fbd5e5e4899.pdf)
