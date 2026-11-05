---
type: entity
project_id: primesim
tags:
  - paper
  - world-model
  - control
status: complete
updated: 2026-11-05
arxiv: "2611.04870"
project: https://primebottech.github.io/PrimeSim-website/
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/primesim_arxiv_2611_04870.md
  - ../../sources/sites/primesim.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "PrimeSim（arXiv:2611.04870）：PrimeSim 用跨平台示范对齐动作与视频帧，再让多个视角交换信息。"
---

# PrimeSim（arXiv:2611.04870）

**PrimeSim**（*PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations*，[arXiv:2611.04870](https://arxiv.org/abs/2611.04870)，[项目页](https://primebottech.github.io/PrimeSim-website/)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「世界模型与控制」组。

## 一句话定义

**PrimeSim 用跨平台示范对齐动作与视频帧，再让多个视角交换信息。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PrimeSim | PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations | 本文方法 / 基准的简称 |

## 为什么重要

- 让不同视角的模拟相机看见同一次动作。它在世界模型测试中能维持较长的连续预测，但细小位移的局部校准仍落后，使用时值得单独检验。
- 开源结论：**部分开源**（步骤 2.5，2026-11-05）。公众号未给代码链接；项目页链接的 PrimebotTech/PrimeSim 经 git ls-remote 可达，是否即论文完整代码未核对。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04870](https://arxiv.org/abs/2611.04870) |
| **开源** | **部分开源** |
| **要点** | PrimeSim 用跨平台示范对齐动作与视频帧，再让多个视角交换信息。 |
| **文内指标** | 公众号转述：它在世界模型测试中能维持较长的连续预测，但细小位移的局部校准仍落后，使用时值得单独检验。 |

## 源码运行时序图

**部分开源** — 公众号未给代码链接；项目页链接的 PrimebotTech/PrimeSim 经 git ls-remote 可达，是否即论文完整代码未核对。 待可运行代码公开后补图。

## 实验与评测

- 公众号转述：它在世界模型测试中能维持较长的连续预测，但细小位移的局部校准仍落后，使用时值得单独检验。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让不同视角的模拟相机看见同一次动作：它在世界模型测试中能维持较长的连续预测，但细小位移的局部校准仍落后，使用时值得单独检验。**

1. 开源边界：**部分开源** — 公众号未给代码链接；项目页链接的 PrimebotTech/PrimeSim 经 git ls-remote 可达，是否即论文完整代码未核对。
2. 核心机制：PrimeSim 用跨平台示范对齐动作与视频帧，再让多个视角交换信息。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [primesim_arxiv_2611_04870.md](../../sources/papers/primesim_arxiv_2611_04870.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.04870](https://arxiv.org/abs/2611.04870)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04870)
- [项目页](https://primebottech.github.io/PrimeSim-website/)
