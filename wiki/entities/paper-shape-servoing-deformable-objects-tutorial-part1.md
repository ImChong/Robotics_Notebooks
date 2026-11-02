---
type: entity
tags:
  - paper
  - deformable-manipulation
  - manipulation
status: complete
updated: 2026-11-02
arxiv: "2610.40559"
project_id: shape-servoing-deformable-objects-tutorial-part1
related:
  - ../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/shape-servoing-deformable-objects-tutorial-part1_arxiv_2610_40559.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
summary: "可变形物体形状伺服教程（一）：基础原理（arXiv:2610.40559）：系统介绍基于形变雅可比矩阵的形状伺服方法，包括形状表示、离线估计、在线 Broyden 更新、闭环控制律及局部稳定性分析。使用双臂操作弹性杆的仿真实例解释完整控制流程。"
---

# 可变形物体形状伺服教程（一）：基础原理

**形状伺服教程 Part I**（*A Tutorial on Shape Servoing of Deformable Objects-Part I: Fundamentals*，[arXiv:2610.40559](https://arxiv.org/abs/2610.40559)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**可变形物体形状伺服教程（一）：基础原理——系统介绍基于形变雅可比矩阵的形状伺服方法，包括形状表示、离线估计、在线 Broyden 更新、闭环控制律及局部稳定性分析。使用双臂操作弹性杆的仿真实例解释完整控制流程。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：机器人控制柔性物体达到目标形状时，需要知道机器人运动如何影响物体形变，但该关系依赖材料、几何和抓取边界条件，难以准确建模。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.40559](https://arxiv.org/abs/2610.40559) · [PDF](https://arxiv.org/pdf/2610.40559) |
| **作者（arXiv）** | Cuiral-Zueco, Ignacio, Caporali, Alessio, Aranda, Miguel, Palli, Gianluca, López-Nicolás, Gonzalo, Mezouar, Youcef |
| **机构（周报所述，未逐条核实）** | 萨拉戈萨大学、博洛尼亚大学、Clermont Auvergne INP。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 系统介绍基于形变雅可比矩阵的形状伺服方法，包括形状表示、离线估计、在线 Broyden 更新、闭环控制律及局部稳定性分析。使用双臂操作弹性杆的仿真实例解释完整控制流程。 |

### arXiv 摘要节选（英文原文）

> In robotics, shape control (or shape servoing) refers to the automatic manipulation of a deformable object to achieve a desired shape. It involves a feedback loop that exploits the relation between robot motion and changes in the measured shape.

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

**可变形物体形状伺服教程（一）：基础原理。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-manipulation-technology-map](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [shape-servoing-deformable-objects-tutorial-part1_arxiv_2610_40559.md](../../sources/papers/shape-servoing-deformable-objects-tutorial-part1_arxiv_2610_40559.md)
- [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
- [arXiv:2610.40559](https://arxiv.org/abs/2610.40559)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.40559)
