---
type: entity
project_id: anchorhec
tags:
  - paper
  - world-model
  - perception
status: complete
updated: 2026-11-04
arxiv: "2611.02884"
project: https://hwanggb.github.io/AnchorHEC/
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/anchorhec_arxiv_2611_02884.md
  - ../../sources/sites/anchorhec.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "AnchorHEC（arXiv:2611.02884）：结合RGB重建和正运动学，先精化腕相机，再用度量场景点定位固定相机；围绕无标靶联合标定处理重建尺度误差，并保留物理机器人验证。"
---

# AnchorHEC（arXiv:2611.02884）

**AnchorHEC**（*AnchorHEC: Targetless Hand-Eye Calibration of Wrist and Fixed Cameras*，[arXiv:2611.02884](https://arxiv.org/abs/2611.02884)，[项目页](https://hwanggb.github.io/AnchorHEC/)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「世界模型与感知」组。

## 一句话定义

**结合RGB重建和正运动学，先精化腕相机，再用度量场景点定位固定相机；围绕无标靶联合标定处理重建尺度误差，并保留物理机器人验证。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AnchorHEC | AnchorHEC: Targetless Hand-Eye Calibration of Wrist and Fixed Cameras | 本文方法 / 基准的简称 |

## 为什么重要

- 不放标定板，校准腕相机与固定相机。
- 开源结论：**待发布**（步骤 2.5，2026-11-04）。项目页 Code 按钮标 coming soon（disabled）；页面另链接 HwangGB/AnchorHEC，git ls-remote 可达，是否已含论文代码以仓库 README 为准。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.02884](https://arxiv.org/abs/2611.02884) |
| **开源** | **待发布** |
| **要点** | 结合RGB重建和正运动学，先精化腕相机，再用度量场景点定位固定相机；围绕无标靶联合标定处理重建尺度误差，并保留物理机器人验证。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 Code 按钮标 coming soon（disabled）；页面另链接 HwangGB/AnchorHEC，git ls-remote 可达，是否已含论文代码以仓库 README 为准。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**不放标定板，校准腕相机与固定相机：结合RGB重建和正运动学，先精化腕相机，再用度量场景点定位固定相机；围绕无标靶联合标定处理重建尺度误差，并保留物理机器人验证。**

1. 开源边界：**待发布** — 项目页 Code 按钮标 coming soon（disabled）；页面另链接 HwangGB/AnchorHEC，git ls-remote 可达，是否已含论文代码以仓库 README 为准。
2. 核心机制：结合RGB重建和正运动学，先精化腕相机，再用度量场景点定位固定相机；围绕无标靶联合标定处理重建尺度误差，并保留物理机器人验证。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [anchorhec_arxiv_2611_02884.md](../../sources/papers/anchorhec_arxiv_2611_02884.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.02884](https://arxiv.org/abs/2611.02884)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.02884)
- [项目页](https://hwanggb.github.io/AnchorHEC/)
