---
type: entity
tags:
  - paper
  - cross-embodiment
  - manipulation
status: complete
updated: 2026-11-02
arxiv: "2610.39167"
project_id: egocontact-egocentric-physical-intelligence
related:
  - ../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/egocontact-egocentric-physical-intelligence_arxiv_2610_39167.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
summary: "EgoContact：利用第一视角视频评估物理智能（arXiv:2610.39167）：构建包含 22,500 个人工验证交互事件的数据集，提供接触前观测、操作意图和像素级接触位置。将 EgoContact 作为可供性迁移策略的记忆库后，真实机器人抓取成功率从 **16% 提升至 58%"
---

# EgoContact：利用第一视角视频评估物理智能

**EgoContact**（*EgoContact: Evaluating Physical Intelligence with Egocentric Video*，[arXiv:2610.39167](https://arxiv.org/abs/2610.39167)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**EgoContact：利用第一视角视频评估物理智能——构建包含 22,500 个人工验证交互事件的数据集，提供接触前观测、操作意图和像素级接触位置。将 EgoContact 作为可供性迁移策略的记忆库后，真实机器人抓取成功率从 **16% 提升至 58%**。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：人类第一视角视频被广泛用于机器人预训练，但模型是否真正理解物体应该在哪里接触，缺少直接评测。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.39167](https://arxiv.org/abs/2610.39167) · [PDF](https://arxiv.org/pdf/2610.39167) |
| **作者（arXiv）** | Deryło, Adam, Gavryushin, Alexey, Cremers, Daniel, Pollefeys, Marc, Wang, Xi |
| **机构（周报所述，未逐条核实）** | 慕尼黑工业大学、苏黎世联邦理工学院、剑桥大学等。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 构建包含 22,500 个人工验证交互事件的数据集，提供接触前观测、操作意图和像素级接触位置。将 EgoContact 作为可供性迁移策略的记忆库后，真实机器人抓取成功率从 **16% 提升至 58%**。 |

### arXiv 摘要节选（英文原文）

> Egocentric video has emerged as one of the most scalable sources of data for learning generalizable manipulation priors. Yet, despite its growing use in pretraining robotic policies, it has been underutilized as a validation tool for physical intelligence, a key component of which is understanding where to make contact with an object to achieve an intended interaction.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**EgoContact：利用第一视角视频评估物理智能。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-manipulation-technology-map](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [egocontact-egocentric-physical-intelligence_arxiv_2610_39167.md](../../sources/papers/egocontact-egocentric-physical-intelligence_arxiv_2610_39167.md)
- [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
- [arXiv:2610.39167](https://arxiv.org/abs/2610.39167)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.39167)
