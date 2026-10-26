---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-10-26
arxiv: "2610.27688"
project_id: adaptive-traversal-tubes-bipedal-navigation
code: https://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces
related:
  - ../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/adaptive-traversal-tubes-bipedal-navigation_arxiv_2610_27688.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
summary: "Traversal Tubes：面向双足人形姿态感知导航的自适应通行管道（arXiv:2610.27688）：用沿路径变化的通行宽度统一表示几何净空和社交距离，再将其映射成正向—侧向行走姿态及全向速度；配套训练侧步增强的 AMP 运动先验。当前版本的定量评测仍待补充。"
---

# Traversal Tubes：面向双足人形姿态感知导航的自适应通行管道

**Traversal Tubes**（*Adaptive Traversal Tubes for Posture-Aware Navigation of Bipedal Humanoids*，[arXiv:2610.27688](https://arxiv.org/abs/2610.27688)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**Traversal Tubes：面向双足人形姿态感知导航的自适应通行管道——用沿路径变化的通行宽度统一表示几何净空和社交距离，再将其映射成正向—侧向行走姿态及全向速度；配套训练侧步增强的 AMP 运动先验。当前版本的定量评测仍待补充。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：传统导航只规划路径和速度，无法主动利用侧身姿态穿越窄通道或为行人让路。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.27688](https://arxiv.org/abs/2610.27688) · [PDF](https://arxiv.org/pdf/2610.27688) |
| **作者（arXiv）** | Zijian Ge |
| **机构（周报所述，未逐条核实）** | 作者未注明机构 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页/评论给出 GitHub 链接（http://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces；仅核对到链接存在，未核验仓库内容）（2026-10-26） |
| **周报所述创新点** | 用沿路径变化的通行宽度统一表示几何净空和社交距离，再将其映射成正向—侧向行走姿态及全向速度；配套训练侧步增强的 AMP 运动先验。当前版本的定量评测仍待补充。 |

### arXiv 摘要节选（英文原文）

> Bipedal humanoids can change their effective footprint by adjusting body yaw, enabling both narrow-passage traversal and socially aware yielding. This paper presents a posture-aware navigation framework centered on \emph{Adaptive Traversal Tubes} (ATT), \emph{Interval Velocity--Heading} (IVH) control, and a sidestep-enhanced AMP locomotion prior.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出 GitHub 链接（http://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces；仅核对到链接存在，未核验仓库内容），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出 GitHub 链接（http://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces；仅核对到链接存在，未核验仓库内容） |

## 结论

**Traversal Tubes：面向双足人形姿态感知导航的自适应通行管道。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [adaptive-traversal-tubes-bipedal-navigation_arxiv_2610_27688.md](../../sources/papers/adaptive-traversal-tubes-bipedal-navigation_arxiv_2610_27688.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [arXiv:2610.27688](https://arxiv.org/abs/2610.27688)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.27688)
