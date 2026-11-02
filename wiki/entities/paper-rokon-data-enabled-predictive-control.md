---
type: entity
tags:
  - paper
  - imitation-learning
  - manipulation
status: complete
updated: 2026-11-02
arxiv: "2610.35458"
project_id: rokon-data-enabled-predictive-control
related:
  - ../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/rokon-data-enabled-predictive-control_arxiv_2610_35458.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
summary: "RoKON：面向机器人操作的高效自适应数据驱动预测控制（arXiv:2610.35458）：使用高斯核构建非线性数据预测器，并提出阻尼 Gauss–Newton 优化方法提高实时求解效率。通过混合数据缓存同时保留有用的离线经验和新采集的在线数据，使控制器能够适应变化的接触动力学。"
---

# RoKON：面向机器人操作的高效自适应数据驱动预测控制

**RoKON**（*RoKON: Efficient and Adaptive Data-Enabled Predictive Control for Robotic Manipulation*，[arXiv:2610.35458](https://arxiv.org/abs/2610.35458)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**RoKON：面向机器人操作的高效自适应数据驱动预测控制——使用高斯核构建非线性数据预测器，并提出阻尼 Gauss–Newton 优化方法提高实时求解效率。通过混合数据缓存同时保留有用的离线经验和新采集的在线数据，使控制器能够适应变化的接触动力学。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：机器人操作中的摩擦等动力学参数可能随环境变化。依赖固定动力学模型的预测控制难以及时适应这些变化。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.35458](https://arxiv.org/abs/2610.35458) · [PDF](https://arxiv.org/pdf/2610.35458) |
| **作者（arXiv）** | Sharma, Akshun, Wei, Zhongqi, Ou, Buqing, Dümbgen, Frederike |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Akshun Sharma、Zhongqi Wei 等）。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 使用高斯核构建非线性数据预测器，并提出阻尼 Gauss–Newton 优化方法提高实时求解效率。通过混合数据缓存同时保留有用的离线经验和新采集的在线数据，使控制器能够适应变化的接触动力学。 |

### arXiv 摘要节选（英文原文）

> Data-enabled predictive control has recently emerged as a promising control paradigm, enabling a transparent optimal-control formulation without the need to learn an explicit dynamics model. This paper presents a number of extensions paving the path towards its use in real-time robotics tasks.

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

**RoKON：面向机器人操作的高效自适应数据驱动预测控制。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-manipulation-technology-map](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [rokon-data-enabled-predictive-control_arxiv_2610_35458.md](../../sources/papers/rokon-data-enabled-predictive-control_arxiv_2610_35458.md)
- [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
- [arXiv:2610.35458](https://arxiv.org/abs/2610.35458)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.35458)
