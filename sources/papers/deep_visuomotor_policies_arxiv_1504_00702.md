# End-to-End Training of Deep Visuomotor Policies（arXiv:1504.00702）

- **类型：** paper / first-party original reading
- **标题：** End-to-End Training of Deep Visuomotor Policies
- **作者：** Sergey Levine、Chelsea Finn、Trevor Darrell、Pieter Abbeel
- **原文：** https://arxiv.org/abs/1504.00702v5
- **HTML：** https://arxiv.org/html/1504.00702v5
- **PDF：** https://arxiv.org/pdf/1504.00702v5
- **核对范围：** 2026-11-09 阅读 arXiv v5 §1–7、附录 A.1–A.4、B.1–B.2、实验表和图注；未独立复现原实验。
- **通读：** 已完成
- **一句话说明：** 训练期 full state 的局部优化通过 BADMM Guided Policy Search 监督视觉策略。

## 原文方法和评测核对

局部线性高斯动力学拟合与 KL 约束轨迹优化，BADMM 协调局部策略与 CNN 监督；视觉 CNN 约92k参数，spatial softmax 形成可解释位置特征。真实 PR2 测试含挂衣架、形状分类、玩具锤子、拧瓶盖；Table 4 的156/171/240/288是各任务测试轨迹数量而非成功率。总体训练墙钟约3–4小时，物理采样仅一部分。附录B.2明确 PR2 effort 接口设置前馈电机电压，近似力矩但不具备闭环力矩控制。

## 局限与复现边界

cbfinn/gps 是相关GPS算法重实现，不能代表原论文全部实验源码。不同视觉干扰下泛化能力受限制。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-deep-visuomotor-policies.md)
- [11 篇阅读路线](../../wiki/overview/robot-learning-foundations-11-papers.md)
