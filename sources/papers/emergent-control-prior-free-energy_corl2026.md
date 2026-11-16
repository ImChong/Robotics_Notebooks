# Emergent Control via Prior-Based Free Energy Minimization（CoRL 2026）

> 来源归档（paper）

- **标题：** Emergent Control via Prior-Based Free Energy Minimization
- **类型：** paper
- **作者：** Manish Kolachalam, Rani Malhotra（Papers with Code 镜像页所列）
- **会场：** CoRL 2026（Spotlight）
- **OpenReview：** <https://openreview.net/forum?id=XY8QpMSxvq>（本环境被验证挑战挡住，403）
- **arXiv：** 未搜到（export.arxiv.org 标题/作者检索无结果）
- **镜像页：** <https://paperswithcode.co/paper/132867>（仅摘要与元数据）
- **入库日期：** 2026-11-12
- **一句话说明：** 以「偏离偏好状态先验的程度」（自由能）替代奖励最大化的控制框架，含基于世界模型的规划器与无模型控制器两部分（据摘要）。

## 开源状态

- 未在 Papers with Code 镜像页与 CoRL explorer 条目中找到代码、项目页或 PDF 链接（2026-11-12 检索）。

## 核对范围与通读

- **核对范围：** 仅读到摘要级信息（Papers with Code 镜像页的摘要转述、operatic-ai/corl-explorer 条目的 TLDR、Berseth 阅读清单一句话摘要）；论文正文、方法公式、实验表、附录均未读到，OpenReview PDF 返回 403，arXiv 无此条目。
- **通读：** 未完成

## 核心摘录（均为摘要级，作者自述）

1. 智能体通过最小化相对「偏好状态先验」的偏离来行动，而非最大化奖励。
2. 两部分：基于学习世界模型与轨迹优化的模型型自由能规划器；用深度策略优化训练的无模型控制器。
3. 报告在连续运动、自动驾驶、欠驱动控制任务上出现连贯自适应行为，「无需环境定义的显式奖励」；稀疏奖励设定下仅靠偏好状态先验学到目标导向行为；稳定步态来自稳态（homeostatic）与运动学先验；规划器产生避碰、求生存的驾驶行为。
4. 以上未经正文核对：是否真机、仿真平台、基线、数值均不知。

## 对 wiki 的映射

- [paper-emergent-control-prior-free-energy](../../wiki/entities/paper-emergent-control-prior-free-energy.md)（draft）
