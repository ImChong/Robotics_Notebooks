---
type: entity
project_id: r2-wam-visual-representations
tags: [paper, world-action-models, wam, visual-representation, robot-manipulation, casia, galbot, pku, ustc]
status: complete
updated: 2026-10-20
arxiv: "2610.13024"
project: https://kailiu18.github.io/R-2WAM.github.io/
related:
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ./paper-wam-design-empirical-study.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/r2_wam_visual_representations_arxiv_2610_13024.md
  - ../../sources/sites/r2-wam-visual-representations-project.md
summary: "R²-WAM 在统一训练设置下比较 13 种视觉表征与 5 种集成机制，发现强通用表征和通道维联合预测是更可靠的起点；多源组合须按任务增量筛选。"
---

# R²-WAM：世界动作模型需要什么视觉表征？

**R²-WAM** 对 WAM 的视觉输入做受控比较：固定训练设置，分别改变预训练视觉表征、特征集成机制和多源组合，研究这些选择如何影响机器人操作策略。

## 一句话定义

R²-WAM 是一项受控实证研究及其实例化配方，用来回答 WAM 应使用哪些视觉特征、又应怎样把它们接入预测与动作策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 同一策略内联合建模世界演化与可执行动作 |
| VAE | Variational Autoencoder | 视频生成骨干所用、以重建为主要目标的潜表示 |
| R²-WAM | Representation Recipe for World Action Models | 论文给出的视觉表征组合配方 |
| DINOv3 | DINO Version 3 | R²-WAM 组合中的通用视觉表征来源 |
| V-JEPA | Video Joint-Embedding Predictive Architecture | 提供视频表征的自监督预测模型系列 |

## 为什么重要

视频生成式 WAM 的 VAE 潜变量继承了强时空先验，但其训练目标偏向重建视频。机器人操作还需要视觉特征保留与任务相关的物体语义、空间关系和交互线索。若同时换表征与接入方式，就很难判断收益来自哪一部分；这篇工作通过受控实验把二者拆开比较。

## 核心原理

论文围绕三个问题组织实验：

1. **RQ1：哪些视觉来源有用？** 在共同的训练协议下比较 13 种预训练表征，覆盖通用视觉与特定视觉能力方向。
2. **RQ2：怎样接入 WAM？** 比较 5 种特征集成机制，并检查附加视觉信息是否真正进入未来预测和动作专家的计算。
3. **RQ3：哪些来源适合组合？** 评估多源视觉特征组合，关注新增来源相对已有表征提供的任务相关信息，而非单独表征排名。

论文实例化的 R²-WAM 组合使用 DINOv3、UniTok 与 V-JEPA 2.1，并采用通道维联合预测。其经验顺序是先选择强通用视觉表征，再验证接入机制，最后按增量价值决定是否增加表征来源。

## 实验与评测

- 论文在统一设置下开展表征、集成和组合比较，并报告机器人操作任务结果。
- R²-WAM 在 **RoboCasa GR1 Tabletop** 上达到 **84.7%** 成功率，相对 VAE-only 基线提升 **5.9 个百分点**。
- 在 **RoboTwin 2.0** 上达到 **92.8%**，相对 VAE-only 基线提升 **2.2 个百分点**。
- 摘要报告通道维联合预测取得最高平均成功率；项目页面提供论文图表与数据交互展示，细分协议和统计应以论文正文为准。

## 与其他工作对比

| 设计选择 | 论文观察 | 选型含义 |
|----------|----------|----------|
| 通用或专用视觉表征 | 强通用表征提供较广鲁棒性收益，可匹配或超过特定能力表征 | 优先建立通用表征基线，再验证任务专用信号的增量 |
| 特征集成机制 | 通道维联合预测取得最高平均成功率 | 比较表征时应固定接入机制，避免把两个因素混在一起 |
| 多源特征组合 | 单一来源的作用会随已有组合变化 | 组合价值应看条件增益，避免按单模态得分简单叠加 |

相较于 [WAM 设计要素受控实证](./paper-wam-design-empirical-study.md)，本研究重点放在视觉来源与注入方式；相较于 [ACG-WAM](./paper-acg-wam-geometric-latent-prediction.md)，它提供广泛表征选择证据，而非只加入一种几何辅助预测目标。

## 工程实践与局限

- **建议的消融顺序：** 固定骨干与数据协议；先比较视觉来源，再比较接入机制，最后测试组合。每次只改变一个维度，便于定位收益。
- **组合评估：** 记录加入来源前后的成功率、鲁棒性和计算开销；来源之间可能冗余或互相干扰。
- **证据边界：** 论文摘要显示 RoboCasa 和 RoboTwin 上的结果；跨本体与其它操作场景的适用性仍需单独验证。详细评估设置以论文正文为准。
- **源码运行时序图：** 不适用。项目页提供静态展示网站及其源码，未提供可运行的研究训练、推理或部署实现，因此不能据此绘制研究代码执行路径。
- **开放状态：** 截至 2026-10-20，项目页和论文未列官方研究代码、权重或数据集下载链接；公开的 GitHub 仓库是网站源码。

## 结论

**可靠起点是强通用视觉表征与有效的联合预测机制；多源组合只有在提供新增任务信息时才值得承担额外复杂度。**

1. 先建立 VAE-only 基线，再优先试强通用视觉特征。
2. 表征对比时固定集成方法；通道维联合预测是论文报告的最佳平均机制。
3. 不按单个编码器的独立表现推断组合价值，要测量相对现有组合的增量。
4. 84.7% 与 92.8% 是特定 RoboCasa / RoboTwin 设置下的结果，不能当作跨任务保证。
5. 当前开放材料是项目展示站；研究实现尚未公开，复现时需自行实现或等待官方发布。

## 关联页面

- [World Action Models（WAM）](../concepts/world-action-models.md) — WAM 的任务边界与架构族谱
- [WAM 设计要素受控实证](./paper-wam-design-empirical-study.md) — 视频–动作因果、潜空间与训练目标的消融
- [ACG-WAM](./paper-acg-wam-geometric-latent-prediction.md) — 训练期加入动作条件几何预测目标
- [Manipulation](../tasks/manipulation.md) — 操作策略常见评测任务

## 参考来源

- [论文摘录](../../sources/papers/r2_wam_visual_representations_arxiv_2610_13024.md)
- [项目页核查](../../sources/sites/r2-wam-visual-representations-project.md)

## 推荐继续阅读

- [arXiv:2610.13024](https://arxiv.org/abs/2610.13024)
- [R²-WAM 项目页](https://kailiu18.github.io/R-2WAM.github.io/)
