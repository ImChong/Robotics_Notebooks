---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-11-02
arxiv: "2610.37974"
project_id: forge-humanoid-reduced-order-reference
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/forge-humanoid-reduced-order-reference_arxiv_2610_37974.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行（arXiv:2610.37974）：先在单刚体动力学和完整运动学组成的降阶环境中训练参考生成器，再用其在线输出训练全阶跟踪策略；单一系统覆盖前后、侧向、转向、单腿跳和连续技能转换。"
---

# FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行

**FORGE**（*FORGE: Feasible Online Reduced-Order Reference Generation and Whole-Body Execution for Agile Humanoid Locomotion*，[arXiv:2610.37974](https://arxiv.org/abs/2610.37974)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行——先在单刚体动力学和完整运动学组成的降阶环境中训练参考生成器，再用其在线输出训练全阶跟踪策略；单一系统覆盖前后、侧向、转向、单腿跳和连续技能转换。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：降阶动力学模型可快速生成动作，但生成的参考不一定满足完整人形机器人的运动学和执行器约束。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37974](https://arxiv.org/abs/2610.37974) · [PDF](https://arxiv.org/pdf/2610.37974) |
| **作者（arXiv）** | Ghansah, Adrian B., An, Yujin, Kim, Jeeseop, Chung, Soon-Jo |
| **机构（周报所述，未逐条核实）** | 加州理工学院、得克萨斯大学埃尔帕索分校 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 先在单刚体动力学和完整运动学组成的降阶环境中训练参考生成器，再用其在线输出训练全阶跟踪策略；单一系统覆盖前后、侧向、转向、单腿跳和连续技能转换。 |

### arXiv 摘要节选（英文原文）

> This paper presents a two-stage learning framework for versatile and agile humanoid locomotion that couples online reduced-order model (ROM) reference generation with whole-body tracking control. In the first stage, we train a command-conditioned whole-body reference generator using single rigid body dynamics combined with full humanoid kinematics.

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

**FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [forge-humanoid-reduced-order-reference_arxiv_2610_37974.md](../../sources/papers/forge-humanoid-reduced-order-reference_arxiv_2610_37974.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.37974](https://arxiv.org/abs/2610.37974)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37974)
