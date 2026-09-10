---
type: entity
tags: [paper, world-models, latent-planning, multi-view, manipulation]
status: complete
updated: 2026-09-10
arxiv: "2609.10506"
code: https://github.com/utn-air/DUET-DINO
related:
  - ../methods/imitation-learning.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/vlm-manipulation-11-papers-technology-map.md
sources:
  - ../../sources/papers/duet-dino_arxiv_2609_10506.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md
summary: "静态侧视+腕部相机联合预测动作条件未来表征，CEM 在 7-DoF 空间做 latent planning；RoboLab reach 92%、angled-reach 72.5%、grasp-and-lift 60%。"
---

# DUET-DINO（arXiv:2609.10506）

**DUET-DINO**（[DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation](https://arxiv.org/abs/2609.10506)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)。静态侧视+腕部相机联合预测动作条件未来表征，CEM 在 7-DoF 空间做 latent planning；RoboLab reach 92%、angled-reach 72.5%、grasp-and-lift 60%。

## 一句话定义

**RoboLab 三任务成功率 92% / 72.5% / 60.0%；CEM 每步大量候选，作者指出限制实时性。**

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
| **arXiv** | [2609.10506](https://arxiv.org/abs/2609.10506) |
| **项目页** | https://utn-air.github.io/DUET-DINO |
| **代码** | https://github.com/utn-air/DUET-DINO |
| **开源** | **待发布** |
| **文内指标** | RoboLab 三任务成功率 92% / 72.5% / 60.0%；CEM 每步大量候选，作者指出限制实时性。 |


## 源码运行时序图

**不适用**（截至入库日 GitHub 仓仅为学术主页模板，无可运行训练/规划代码。）。


## 结论

**DUET-DINO 值得按「待发布」边界阅读：先核对仓库是否可跑，再引用文内成功率数字。**

1. 索引来源为公众号导读，实验细节以 arXiv PDF 为准。
2. 开源结论：**待发布** — 截至入库日 GitHub 仓仅为学术主页模板，无可运行训练/规划代码。。
3. 选型时对照 [11 篇地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中相邻节点，避免重复造页。

## 关联页面

- [VLM 与操作 11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md)
- [模仿学习 (Imitation Learning)](../methods/imitation-learning.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [duet-dino_arxiv_2609_10506.md](../../sources/papers/duet-dino_arxiv_2609_10506.md)
- [wechat 11篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- [arXiv:2609.10506](https://arxiv.org/abs/2609.10506)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.10506)
- [项目页](https://utn-air.github.io/DUET-DINO)
- [GitHub](https://github.com/utn-air/DUET-DINO)
