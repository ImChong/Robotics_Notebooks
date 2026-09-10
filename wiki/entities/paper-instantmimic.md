---
type: entity
tags: [paper, reinforcement-learning, gpu-training, motion-tracking, humanoid]
status: complete
updated: 2026-09-10
arxiv: "2609.09821"
code: https://github.com/Scripter36/InstantMimic
related:
  - ../methods/imitation-learning.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/vlm-manipulation-11-papers-technology-map.md
sources:
  - ../../sources/papers/instantmimic_arxiv_2609_09821.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md
summary: "GPU-native 训练环整合仿真/环境/策略/更新；标准动作跟踪秒级收敛，37.4h AMASS 预训练压到约 30 分钟。"
---

# InstantMimic（arXiv:2609.09821）

**InstantMimic**（[InstantMimic: A High Performance System for Learning Physics-based Skills in Seconds](https://arxiv.org/abs/2609.09821)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)。GPU-native 训练环整合仿真/环境/策略/更新；标准动作跟踪秒级收敛，37.4h AMASS 预训练压到约 30 分钟。

## 一句话定义

**强调 kernel 碎片化与 CPU 内存访问是 RL 瓶颈；官方 README 写 Code will be released soon。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 从专家示范学习策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| RL | Reinforcement Learning | 强化学习 |
| CEM | Cross-Entropy Method | 采样优化动作/轨迹的规划器 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 纳入本期 **VLM 控制 / 世界模型 / 灵巧操作 / 规划 / 评测** 主线之一。
- 开源状态：**待发布**（步骤 2.5 核查，2026-09-10）。
- 与 [11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.09821](https://arxiv.org/abs/2609.09821) |
| **项目页** | https://scripter36.github.io/projects/instantmimic/ |
| **代码** | https://github.com/Scripter36/InstantMimic |
| **开源** | **待发布** |
| **文内指标** | 强调 kernel 碎片化与 CPU 内存访问是 RL 瓶颈；官方 README 写 Code will be released soon。 |


## 源码运行时序图

**不适用**（GitHub 仓存在但 README 仅占位「即将发布」。）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 标准动作跟踪 | **秒级** 收敛 |
| AMASS 预训练 | **37.4 h → 约 30 min** |
| 归因 | **kernel 碎片化** 与 **CPU 内存访问** 是 RL 训练的主瓶颈 |
| 做法 | GPU-native 训练环：仿真 / 环境 / 策略 / 更新 **全部整合在 GPU 上** |

- **量的是系统不是算法：** 本文报告的是 **wall-clock 与吞吐**，不是新的成功率上限；引用时不要与「跟踪质量更好」混为一谈。
- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md) 与项目页；硬件配置、基线实现与 batch 设定以 **原文 PDF** 为准（[参考来源](#参考来源)）——跨论文的加速比不可脱离硬件横比。
- **复现边界：** 官方 README 写 *Code will be released soon*，截至 **2026-09-10** 无可跑入口。

## 与其他工作对比

| 对照路线 | 差异 |
|----------|------|
| GPU 仿真 + CPU 侧 RL 循环（常规 Isaac / MJX 训练栈） | 仿真在 GPU、环境包装与更新回到 CPU，逐步引入 **数据搬运与 kernel launch** 开销；本文把整环留在 GPU。 |
| 增大并行环境数 | 靠 worlds 数量摊薄开销，不消除碎片化；本文归因到 **kernel 碎片化本身**。 |
| 算法侧加速（样本效率、蒸馏、课程） | 减少所需样本量；本文不动算法，减少 **单位样本的时间成本**，两条路可叠。 |
| [SwingBot](./paper-swingbot.md) 等技能学习工作 | 那边问「这个技能能不能学会」，本文问「同一套学习能跑多快」，属正交维度。 |

## 结论

**InstantMimic 值得按「待发布」边界阅读：先核对仓库是否可跑，再引用文内成功率数字。**

1. 索引来源为公众号导读，实验细节以 arXiv PDF 为准。
2. 开源结论：**待发布** — GitHub 仓存在但 README 仅占位「即将发布」。。
3. 选型时对照 [11 篇地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中相邻节点，避免重复造页。

## 关联页面

- [VLM 与操作 11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md)
- [模仿学习 (Imitation Learning)](../methods/imitation-learning.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [instantmimic_arxiv_2609_09821.md](../../sources/papers/instantmimic_arxiv_2609_09821.md)
- [wechat 11篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- [arXiv:2609.09821](https://arxiv.org/abs/2609.09821)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.09821)
- [项目页](https://scripter36.github.io/projects/instantmimic/)
- [GitHub](https://github.com/Scripter36/InstantMimic)
