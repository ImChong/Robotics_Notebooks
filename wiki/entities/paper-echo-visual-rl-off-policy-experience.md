---
type: entity
tags:
  - paper
  - legged-robot
  - locomotion
status: complete
updated: 2026-11-02
arxiv: "2610.37865"
project_id: echo-visual-rl-off-policy-experience
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
  - ../../sources/papers/echo-visual-rl-off-policy-experience_arxiv_2610_37865.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "Echo（离策略视觉 RL）：Echo：利用离策略经验扩展视觉强化学习（arXiv:2610.37865）：让状态输入辅助策略反向模仿持续更新的视觉策略，并采集无需渲染的廉价经验；这些经验通过共享特权 Critic 参与离策略价值学习，直接更新视觉策略。该工作目前为初步仿真实验。"
---

# Echo（离策略视觉 RL）：Echo：利用离策略经验扩展视觉强化学习

**Echo（离策略视觉 RL）**（*Echo: Scaling Visual Reinforcement Learning with Off-Policy Experience*，[arXiv:2610.37865](https://arxiv.org/abs/2610.37865)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**Echo（离策略视觉 RL）：Echo：利用离策略经验扩展视觉强化学习——让状态输入辅助策略反向模仿持续更新的视觉策略，并采集无需渲染的廉价经验；这些经验通过共享特权 Critic 参与离策略价值学习，直接更新视觉策略。该工作目前为初步仿真实验。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：直接视觉 RL 需要持续渲染图像并执行视觉网络，计算成本高；Teacher–Student 两阶段蒸馏又存在模仿误差。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37865](https://arxiv.org/abs/2610.37865) · [PDF](https://arxiv.org/pdf/2610.37865) |
| **作者（arXiv）** | Khandate, Gagan |
| **机构（周报所述，未逐条核实）** | 哥伦比亚大学 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 让状态输入辅助策略反向模仿持续更新的视觉策略，并采集无需渲染的廉价经验；这些经验通过共享特权 Critic 参与离策略价值学习，直接更新视觉策略。该工作目前为初步仿真实验。 |

### arXiv 摘要节选（英文原文）

> Scaling visual reinforcement learning (RL) with GPU-accelerated simulation is computationally expensive due to rendering and visual-policy inference costs. Consequently, two-stage teacher--student distillation is commonly used to train visual policies, but introduces an imitation gap.

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

**Echo（离策略视觉 RL）：Echo：利用离策略经验扩展视觉强化学习。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [locomotion](../tasks/locomotion.md)

## 参考来源

- [echo-visual-rl-off-policy-experience_arxiv_2610_37865.md](../../sources/papers/echo-visual-rl-off-policy-experience_arxiv_2610_37865.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.37865](https://arxiv.org/abs/2610.37865)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37865)
