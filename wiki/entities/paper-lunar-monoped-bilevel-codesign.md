---
type: entity
tags:
  - paper
  - legged-robot
  - locomotion
status: complete
updated: 2026-11-02
arxiv: "2610.37199"
project_id: lunar-monoped-bilevel-codesign
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/lunar-monoped-bilevel-codesign_arxiv_2610_37199.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "月球单足协同设计：面向月球单足机器人二维跳跃与着陆的双层协同设计（arXiv:2610.37199）：外层差分进化搜索腿段长度，内层分别使用 QP 任务空间控制和直接配点轨迹优化评估设计；进一步加入角动量调节和 LQR 落地稳定控制。"
---

# 月球单足协同设计：面向月球单足机器人二维跳跃与着陆的双层协同设计

**月球单足协同设计**（*A Bi-level Co-design Framework for Lunar Monoped with 2D Jump and Landing Control*，[arXiv:2610.37199](https://arxiv.org/abs/2610.37199)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**月球单足协同设计：面向月球单足机器人二维跳跃与着陆的双层协同设计——外层差分进化搜索腿段长度，内层分别使用 QP 任务空间控制和直接配点轨迹优化评估设计；进一步加入角动量调节和 LQR 落地稳定控制。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：月球低重力会延长腾空时间并减小接触力，使腿部尺寸和跳跃控制器高度耦合。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37199](https://arxiv.org/abs/2610.37199) · [PDF](https://arxiv.org/pdf/2610.37199) |
| **作者（arXiv）** | Ramkumar, Sanjay, Kumar, Suraj |
| **机构（周报所述，未逐条核实）** | 印度空间科学与技术学院、U R Rao 卫星中心 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 外层差分进化搜索腿段长度，内层分别使用 QP 任务空间控制和直接配点轨迹优化评估设计；进一步加入角动量调节和 LQR 落地稳定控制。 |

### arXiv 摘要节选（英文原文）

> Legged robots for lunar exploration must be designed for low gravity, where reduced contact forces and long flight phases make leg morphology and controller choice interdependent. We study this for a planar monoped in lunar gravity ($1.625$ m/s$^2$) using a bi-level co-design framework: a differential evolution outer loop searches over link lengths, and each candidate is evaluated on a vertical jump by either a quadratic program based task-space controller (QP-TSC) or a direct-collocation based trajectory optimization.

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

**月球单足协同设计：面向月球单足机器人二维跳跃与着陆的双层协同设计。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [locomotion](../tasks/locomotion.md)

## 参考来源

- [lunar-monoped-bilevel-codesign_arxiv_2610_37199.md](../../sources/papers/lunar-monoped-bilevel-codesign_arxiv_2610_37199.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.37199](https://arxiv.org/abs/2610.37199)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37199)
