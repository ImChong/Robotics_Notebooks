---
type: entity
project_id: tactile-guided-robot-learning
tags: [paper, survey, tactile-sensing, representation-learning, self-supervised-learning, robot-learning]
status: complete
updated: 2026-10-26
arxiv: "2610.20134"
venue: "arXiv preprint (2026)"
related:
  - ../overview/hub-tactile.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../concepts/contact-rich-manipulation.md
  - ./paper-sparsh.md
  - ./paper-tactile-based-robotic-manipulation-survey.md
  - ./paper-tf-art-tactile-force-survey.md
sources:
  - ../../sources/papers/tactile_guided_robot_learning_arxiv_2610_20134.md
  - ../../sources/sites/touch-in-robot-learning-review.md
summary: "这篇综述按触觉在机器人学习管线中的作用，梳理自监督表征、触觉条件策略、基础模型、预测控制与跨任务/传感器泛化；图 2 对比重建、对比和预测式表征目标。"
---

# Tactile-Guided Robot Learning（触觉引导的机器人学习）

**一句话定义：** 这篇综述从学习与控制管线出发，梳理机器人如何把触觉用于表征、动作生成、物理推理、预测控制和泛化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SSL | Self-Supervised Learning | 从无标注交互数据学习触觉表示 |
| VLA | Vision-Language-Action | 将视觉、语言与动作联合建模，可纳入触觉 |
| I-JEPA | Image Joint-Embedding Predictive Architecture | 预测潜在表示而非重建输入外观的代表方法 |
| VBTS | Vision-Based Tactile Sensing | 通过图像读取弹性表面形变的触觉传感 |

## 综述范围

论文按触觉在机器人学习与操作中的功能组织文献，覆盖触觉表示与策略学习、触觉基础模型、预测与自适应控制，以及跨物体、任务、传感器、机器人具身和仿真到真实的泛化。它是领域综述，不提出一个新的机器人策略。

## Figure 2：自监督触觉表征的三类目标

| 范式 | 图中例子 | 训练信号 |
|------|----------|----------|
| 重建式 | Autoencoder、VAE、Masked AE、隐式场 | 从潜变量恢复完整、变换后或被遮挡的触觉观测 |
| 对比式 | 同一传感器增强视图、跨传感器同一接触、同步视触觉、冻结模型锚点 | 让定义为正样本的表示靠近，并与负样本分离 |
| 非对比 / 预测式 | 自蒸馏、潜变量预测、动作条件未来预测 | 对齐教师与学生表示，或从当前上下文预测目标/未来潜变量 |

图中的正样本定义决定模型倾向忽略哪些变化。例如跨传感器对齐关注同一接触的共性，而动作条件预测可以保留接触随操作变化的信息。该图是学习目标的分类示意，不代表三类方法的统一性能比较。

## 证据与阅读方式

综述引用 Sparsh 对 460k 以上无标注触觉观测和六类触觉任务的评测，指出在该基准中 DINO 与 I-JEPA 风格潜变量学习比重建触觉外观的 MAE 显示出更强的迁移表现。这是特定基准上的结果，不等于所有传感器、任务上的通用排序。可对照 [Sparsh](./paper-sparsh.md) 原工作与其评测设置。

阅读时可先检查目标函数保留了什么信息：重建偏向观测保真；对比学习取决于正负样本定义；预测目标则可能编码时序和动作导致的接触变化。再结合传感器类型、下游任务与迁移设置判断结果是否可比。

## 源码运行时序图

**不适用：** 本文是综述，没有独立的训练、推理或部署代码；配套站点提供文献索引，并链接各原始工作的外部资源。

## 与其他综述的关系

| 资料 | 组织视角 | 适合用途 |
|------|----------|----------|
| 本文（arXiv:2610.20134） | 表征学习、策略、基础模型、预测与泛化的学习管线 | 聚焦触觉如何参与机器人学习 |
| [Tactile-Based Robotic Manipulation](./paper-tactile-based-robotic-manipulation-survey.md)（arXiv:2610.30438） | 力学驱动与数据驱动两条研究范式 | 建立触觉操作的总体脉络 |
| [TF-ART](./paper-tf-art-tactile-force-survey.md)（arXiv:2608.07558） | 多模态观测到策略与控制的阶段管线 | 追踪触觉/力觉如何进入控制栈 |

## 结论

**这篇综述的主要价值是把触觉表征、策略、预测和迁移放进同一学习管线，并给出可比较的自监督目标分类。**

1. 选触觉表征时，先明确下游任务要保留的是接触外观、跨传感器共性，还是动作引起的接触变化。
2. 解释对比学习结果时，检查正样本和负样本如何构造；它们定义了表示要忽略的差异。
3. 将 DINO / I-JEPA 与 MAE 的结果限定在论文讨论的 Sparsh 基准，不把它当成跨任务的绝对排名。
4. 复现时沿配套文献库定位原始论文和其代码资源；综述本身没有可运行实现。

## 局限与风险

- 本文提供文献综合与分类，不是使用统一硬件、任务和协议进行的新实验。
- 触觉传感器的数据形态和接触机制差异较大，不同研究的指标不能仅凭模型名称直接比较。
- 配套站点是文献数据库，不是代码仓库；代码可用性应逐篇按原始项目资源核查。

## 关联页面

- [触觉与力觉知识链](../overview/hub-tactile.md) — 触觉表示、传感与控制资料入口
- [触觉传感](../concepts/tactile-sensing.md) — 传感器与触觉量测基础
- [视触觉融合](../concepts/visuo-tactile-fusion.md) — 跨模态表示与信息融合
- [Sparsh](./paper-sparsh.md) — 综述讨论的代表性触觉自监督基准
- [Tactile-Based Robotic Manipulation](./paper-tactile-based-robotic-manipulation-survey.md) — 互补的力学 / 数据驱动分类

## 参考来源

- [论文来源档案](../../sources/papers/tactile_guided_robot_learning_arxiv_2610_20134.md)
- [配套触觉机器人学习文献库](../../sources/sites/touch-in-robot-learning-review.md)

## 推荐继续阅读

- [arXiv 论文与 PDF](https://arxiv.org/abs/2610.20134)
- [作者配套文献库](https://sites.google.com/view/touch-in-robot-learning-review)
