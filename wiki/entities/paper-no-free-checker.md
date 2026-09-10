---
type: entity
tags: [paper, survey, verification, robot-policies, benchmark]
status: complete
updated: 2026-09-10
arxiv: "2609.09250"
code: https://github.com/ZJUSCL/Awesome-Robot-Verifier
related:
  - ../methods/imitation-learning.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/vlm-manipulation-11-papers-technology-map.md
sources:
  - ../../sources/papers/no-free-checker_arxiv_2609_09250.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md
summary: "综述约 150 个 robot policy verifier；人类/规则/学习型/模型内生四类与九项可比指标；ZJUSCL/Awesome-Robot-Verifier 资源索引。"
---

# No Free Checker（arXiv:2609.09250）

**No Free Checker**（[No Free Checker: A Survey of Verifiers for Robot Policies](https://arxiv.org/abs/2609.09250)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)。综述约 150 个 robot policy verifier；人类/规则/学习型/模型内生四类与九项可比指标；ZJUSCL/Awesome-Robot-Verifier 资源索引。

## 一句话定义

**核心贡献是 verifier 分类学与可比性指标，非可运行策略。**

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
- 开源状态：**部分开源**（步骤 2.5 核查，2026-09-10）。
- 与 [11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.09250](https://arxiv.org/abs/2609.09250) |
| **项目页** | — |
| **代码** | https://github.com/ZJUSCL/Awesome-Robot-Verifier |
| **开源** | **部分开源** |
| **文内指标** | 核心贡献是 verifier 分类学与可比性指标，非可运行策略。 |


## 源码运行时序图

**不适用**（配套为论文资源索引仓，非策略训练/评测 harness。）。


## 结论

**No Free Checker 值得按「部分开源」边界阅读：先核对仓库是否可跑，再引用文内成功率数字。**

1. 索引来源为公众号导读，实验细节以 arXiv PDF 为准。
2. 开源结论：**部分开源** — 配套为论文资源索引仓，非策略训练/评测 harness。。
3. 选型时对照 [11 篇地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中相邻节点，避免重复造页。

## 关联页面

- [VLM 与操作 11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md)
- [模仿学习 (Imitation Learning)](../methods/imitation-learning.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [no-free-checker_arxiv_2609_09250.md](../../sources/papers/no-free-checker_arxiv_2609_09250.md)
- [wechat 11篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- [arXiv:2609.09250](https://arxiv.org/abs/2609.09250)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.09250)

- [GitHub](https://github.com/ZJUSCL/Awesome-Robot-Verifier)
