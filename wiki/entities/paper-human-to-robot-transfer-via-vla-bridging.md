---
type: entity
project_id: human-to-robot-transfer-via-vla-bridging
tags:
  - paper
  - data-collection
  - cross-embodiment
status: complete
updated: 2026-11-04
arxiv: "2611.04391"
project: https://junhyeong-park.me/h2rb/
related:
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../tasks/teleoperation.md
  - ../methods/imitation-learning.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/human_to_robot_transfer_via_vla_bridging_arxiv_2611_04391.md
  - ../../sources/sites/human-to-robot-transfer-via-vla-bridging.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Human-to-Robot Transfer via VLA Bridging（arXiv:2611.04391）：同时对齐几何表征与动作潜空间，让未配对的人类和机器人演示共同适配VLA；分离动作解码器，减少跨形态监督对原有控制先验的干扰。"
---

# Human-to-Robot Transfer via VLA Bridging（arXiv:2611.04391）

**Human-to-Robot Transfer via VLA Bridging**（*Human-to-Robot Transfer via Bridging Representation and Action Space in VLAs*，[arXiv:2611.04391](https://arxiv.org/abs/2611.04391)，[项目页](https://junhyeong-park.me/h2rb/)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「数据采集与跨形态迁移」组。

## 一句话定义

**同时对齐几何表征与动作潜空间，让未配对的人类和机器人演示共同适配VLA；分离动作解码器，减少跨形态监督对原有控制先验的干扰。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Human-to-Robot Transfer via VLA Bridging | Human-to-Robot Transfer via Bridging Representation and Action Space in VLAs | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 用人类演示适配机器人策略。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。公众号未给代码链接；项目页链接的 jun-brro/h2rb-code 经 git ls-remote 可达。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04391](https://arxiv.org/abs/2611.04391) |
| **开源** | **已开源** |
| **要点** | 同时对齐几何表征与动作潜空间，让未配对的人类和机器人演示共同适配VLA；分离动作解码器，减少跨形态监督对原有控制先验的干扰。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（已开源）** — 公众号未给代码链接；项目页链接的 jun-brro/h2rb-code 经 git ls-remote 可达。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**用人类演示适配机器人策略：同时对齐几何表征与动作潜空间，让未配对的人类和机器人演示共同适配VLA；分离动作解码器，减少跨形态监督对原有控制先验的干扰。**

1. 开源边界：**已开源** — 公众号未给代码链接；项目页链接的 jun-brro/h2rb-code 经 git ls-remote 可达。
2. 核心机制：同时对齐几何表征与动作潜空间，让未配对的人类和机器人演示共同适配VLA；分离动作解码器，减少跨形态监督对原有控制先验的干扰。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [embodied-data-collection-four-layers-taxonomy](../concepts/embodied-data-collection-four-layers-taxonomy.md)
- [teleoperation](../tasks/teleoperation.md)
- [imitation-learning](../methods/imitation-learning.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [human_to_robot_transfer_via_vla_bridging_arxiv_2611_04391.md](../../sources/papers/human_to_robot_transfer_via_vla_bridging_arxiv_2611_04391.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.04391](https://arxiv.org/abs/2611.04391)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04391)
- [项目页](https://junhyeong-park.me/h2rb/)
