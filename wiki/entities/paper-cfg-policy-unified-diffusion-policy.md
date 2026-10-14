---
type: entity
project_id: cfg-policy-unified-diffusion-policy
tags:
  - paper
  - diffusion-policy
  - classifier-free-guidance
  - reinforcement-learning
status: complete
updated: 2026-10-14
arxiv: "2610.14936"
project: https://zhy-liu.github.io/cfg-policy-page
related:
  - ../methods/diffusion-policy.md
  - ../methods/reinforcement-learning.md
  - ../concepts/diffusion-model.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/cfg_policy_unified_diffusion_policy_arxiv_2610_14936.md
  - ../../sources/sites/cfg-policy-unified-diffusion-policy.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "CFG-Policy（arXiv:2610.14936）：把探索与利用拆成两条生成分支，采样时组合分数/速度场，以减少 RL 微调中的行为多样性损失。"
---

# CFG-Policy（arXiv:2610.14936）

**CFG-Policy**（*CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance*，[arXiv:2610.14936](https://arxiv.org/abs/2610.14936)，[项目页](https://zhy-liu.github.io/cfg-policy-page)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「VLA 与动作生成」组。

## 一句话定义

**把探索与利用拆成两条生成分支，采样时组合分数/速度场，以减少 RL 微调中的行为多样性损失。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CFG | Classifier-Free Guidance | 无分类器引导 |
| DP | Diffusion Policy | 扩散策略 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- 扩散策略 RL 微调常坍缩多样性；CFG 式组合可在采样期调节。
- 开源结论：**待发布**（步骤 2.5，2026-10-14）。项目页含「coming soon」字样，入库日未列代码仓库。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.14936](https://arxiv.org/abs/2610.14936) |
| **开源** | **待发布** |
| **要点** | 在线与离线转在线设置冻结的分支不同。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

**不适用（待发布）** — 项目页含「coming soon」字样，入库日未列代码仓库。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**扩散策略 RL 微调可借 CFG 把探索/利用分离为可调旋钮。**

1. 开源边界：**待发布** — 项目页含「coming soon」字样，入库日未列代码仓库。
2. 核心机制：在线与离线转在线设置冻结的分支不同。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [diffusion-policy](../methods/diffusion-policy.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [diffusion-model](../concepts/diffusion-model.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [cfg_policy_unified_diffusion_policy_arxiv_2610_14936.md](../../sources/papers/cfg_policy_unified_diffusion_policy_arxiv_2610_14936.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.14936](https://arxiv.org/abs/2610.14936)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.14936)
- [项目页](https://zhy-liu.github.io/cfg-policy-page)
