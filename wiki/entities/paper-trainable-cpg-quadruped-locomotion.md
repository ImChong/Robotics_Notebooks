---
type: entity
tags:
  - paper
  - quadruped
  - locomotion
status: complete
updated: 2026-11-02
arxiv: "2610.36537"
project_id: trainable-cpg-quadruped-locomotion
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/trainable-cpg-quadruped-locomotion_arxiv_2610_36537.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "面向四足运动的可训练中央模式发生器框架（arXiv:2610.36537）：八神经元对称 CPG 生成节律，共享腿部执行网络将节律与轨迹参数转换为关节参考；一个反馈策略只调节六个物理可解释变量，即可实现五种步态及全部方向转换。"
---

# 面向四足运动的可训练中央模式发生器框架

**可训练 CPG**（*A Trainable CPG Framework for Quadruped Locomotion*，[arXiv:2610.36537](https://arxiv.org/abs/2610.36537)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**面向四足运动的可训练中央模式发生器框架——八神经元对称 CPG 生成节律，共享腿部执行网络将节律与轨迹参数转换为关节参考；一个反馈策略只调节六个物理可解释变量，即可实现五种步态及全部方向转换。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：多步态控制通常为每种步态和转换分别训练策略，难以获得可解释且连续的统一控制。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.36537](https://arxiv.org/abs/2610.36537) · [PDF](https://arxiv.org/pdf/2610.36537) |
| **作者（arXiv）** | Liu, Xiyan, Liu, Yide, Mao, Guoyong, Qu, Shaoxing |
| **机构（周报所述，未逐条核实）** | 浙江大学 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 八神经元对称 CPG 生成节律，共享腿部执行网络将节律与轨迹参数转换为关节参考；一个反馈策略只调节六个物理可解释变量，即可实现五种步态及全部方向转换。 |

### arXiv 摘要节选（英文原文）

> We present CPG-ENN, a trainable quadruped locomotion framework that combines explicit neural coordination with learned motion realization. A symmetry-based eight-neuron central pattern generator (CPG) supplies gait rhythms, while a pretrained, leg-shared execution neural network (ENN) maps rhythmic signals and trajectory parameters to joint-angle references without online inverse kinematics.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**面向四足运动的可训练中央模式发生器框架。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [locomotion](../tasks/locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [trainable-cpg-quadruped-locomotion_arxiv_2610_36537.md](../../sources/papers/trainable-cpg-quadruped-locomotion_arxiv_2610_36537.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.36537](https://arxiv.org/abs/2610.36537)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.36537)
