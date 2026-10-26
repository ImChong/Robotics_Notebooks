---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-10-26
arxiv: "2610.28925"
project_id: one-policy-many-styles-preference-aggregation
related:
  - ../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/one-policy-many-styles-preference-aggregation_arxiv_2610_28925.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
summary: "One Policy Many Styles：一个策略、多种风格：可引导人形控制应在何处聚合偏好？（arXiv:2610.28925）：建立偏好响应度、覆盖率和效用联合指标，系统比较奖励、优势、损失及 PCGrad 层面的目标聚合；揭示早期聚合响应范围更大，后期聚合控制更精确。"
---

# One Policy Many Styles：一个策略、多种风格：可引导人形控制应在何处聚合偏好？

**One Policy Many Styles**（*One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?*，[arXiv:2610.28925](https://arxiv.org/abs/2610.28925)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**One Policy Many Styles：一个策略、多种风格：可引导人形控制应在何处聚合偏好？——建立偏好响应度、覆盖率和效用联合指标，系统比较奖励、优势、损失及 PCGrad 层面的目标聚合；揭示早期聚合响应范围更大，后期聚合控制更精确。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：偏好条件策略可能虽然响应风格指令，却同时改变无关行为或损害任务性能。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.28925](https://arxiv.org/abs/2610.28925) · [PDF](https://arxiv.org/pdf/2610.28925) |
| **作者（arXiv）** | Ziqian Huang, Yuzhuo Chen, Shimin Zhang, Junyang He, Xinyue Annie Yang, Animesh Garg |
| **机构（周报所述，未逐条核实）** | 佐治亚理工学院 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26） |
| **周报所述创新点** | 建立偏好响应度、覆盖率和效用联合指标，系统比较奖励、优势、损失及 PCGrad 层面的目标聚合；揭示早期聚合响应范围更大，后期聚合控制更精确。 |

### arXiv 摘要节选（英文原文）

> Modern robotic agents are increasingly expected to support steerable control, where a single policy can adjust its behavioral style at runtime under changing user preferences. Preference-conditioned reinforcement learning provides this interface, but conditioning alone does not ensure that a requested behavior change is strong, does not leak into unrelated behaviors, and does not compromise task performance.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**One Policy Many Styles：一个策略、多种风格：可引导人形控制应在何处聚合偏好？。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [one-policy-many-styles-preference-aggregation_arxiv_2610_28925.md](../../sources/papers/one-policy-many-styles-preference-aggregation_arxiv_2610_28925.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [arXiv:2610.28925](https://arxiv.org/abs/2610.28925)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.28925)
