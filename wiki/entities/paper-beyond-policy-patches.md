---
type: entity
project_id: beyond-policy-patches
tags:
  - paper
  - reward-model
  - policy-learning
status: complete
updated: 2026-11-04
arxiv: "2611.02777"
project: https://repair-vla.github.io/
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/beyond_policy_patches_arxiv_2611_02777.md
  - ../../sources/sites/beyond-policy-patches.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Beyond Policy Patches（arXiv:2611.02777）：先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。"
---

# Beyond Policy Patches（arXiv:2611.02777）

**Beyond Policy Patches**（*Beyond Policy Patches: Distilling RL Specialists into Vision-Language-Action Policies*，[arXiv:2611.02777](https://arxiv.org/abs/2611.02777)，[项目页](https://repair-vla.github.io/)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「奖励学习与策略后训练」组。

## 一句话定义

**先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Beyond Policy Patches | Beyond Policy Patches: Distilling RL Specialists into Vision-Language-Action Policies | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 把强化学习专家收进同一个VLA。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-04）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.02777](https://arxiv.org/abs/2611.02777) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把强化学习专家收进同一个VLA：先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [beyond_policy_patches_arxiv_2611_02777.md](../../sources/papers/beyond_policy_patches_arxiv_2611_02777.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.02777](https://arxiv.org/abs/2611.02777)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.02777)
- [项目页](https://repair-vla.github.io/)
