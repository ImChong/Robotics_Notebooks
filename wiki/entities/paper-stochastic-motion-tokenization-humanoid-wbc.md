---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-11-02
arxiv: "2610.36452"
project_id: stochastic-motion-tokenization-humanoid-wbc
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/stochastic-motion-tokenization-humanoid-wbc_arxiv_2610_36452.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "随机动作标记化全身控制：基于随机动作标记化的人形全身控制（arXiv:2610.36452）：用随机动作 Tokenizer 将全身运动压缩成离散 token，并将 token 直接输入跟踪策略；同一接口能够生成并执行行走、格斗、舞蹈以及长时技能转换。"
---

# 随机动作标记化全身控制：基于随机动作标记化的人形全身控制

**随机动作标记化全身控制**（*Whole-Body Humanoid Control via Stochastic Motion Tokenization*，[arXiv:2610.36452](https://arxiv.org/abs/2610.36452)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**随机动作标记化全身控制：基于随机动作标记化的人形全身控制——用随机动作 Tokenizer 将全身运动压缩成离散 token，并将 token 直接输入跟踪策略；同一接口能够生成并执行行走、格斗、舞蹈以及长时技能转换。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：速度指令不足以描述舞蹈、格斗等复杂全身动作，而密集动作参考不适合作为高层任务接口。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.36452](https://arxiv.org/abs/2610.36452) · [PDF](https://arxiv.org/pdf/2610.36452) |
| **作者（arXiv）** | Drolet, Michael, Arenz, Oleg, Peters, Jan |
| **机构（周报所述，未逐条核实）** | 达姆施塔特工业大学、德国人工智能研究中心、hessian.AI、Robotics Institute Germany |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 用随机动作 Tokenizer 将全身运动压缩成离散 token，并将 token 直接输入跟踪策略；同一接口能够生成并执行行走、格斗、舞蹈以及长时技能转换。 |

### arXiv 摘要节选（英文原文）

> While velocity-commanded RL policies are effective for controlling quadrupeds, they are a limited interface for humanoids that need to perform complex whole-body motions. Furthermore, RL tracking policies for whole-body motions typically have limited information about the desired future motion, preventing them from performing controlled deviations to achieve long-term task success; such information is particularly relevant when a controller must prepare for transitions between behaviors.

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

**随机动作标记化全身控制：基于随机动作标记化的人形全身控制。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [stochastic-motion-tokenization-humanoid-wbc_arxiv_2610_36452.md](../../sources/papers/stochastic-motion-tokenization-humanoid-wbc_arxiv_2610_36452.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.36452](https://arxiv.org/abs/2610.36452)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.36452)
