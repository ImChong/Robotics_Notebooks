---
type: entity
project_id: reactworld
tags:
  - paper
  - world-model
  - video-generation
status: complete
updated: 2026-11-06
arxiv: "2611.06120"
project: https://metaengine-world.github.io/ReactWorld/
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-06-20-papers-technology-map.md
sources:
  - ../../sources/papers/reactworld_arxiv_2611_06120.md
  - ../../sources/sites/reactworld.md
  - ../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md
summary: "ReactWorld（arXiv:2611.06120）：模型先推断谁在接触物体、哪些局部会动，再用预测位移引导后续画面。"
---

# ReactWorld（arXiv:2611.06120）

**ReactWorld**（*ReactWorld: Structuring Interaction Responses for Embodied Video Generation*，[arXiv:2611.06120](https://arxiv.org/abs/2611.06120)，[项目页](https://metaengine-world.github.io/ReactWorld/)）来自 [具身智能小站 2026-11-06 盘点](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)，属「交互式世界模型」组。

## 一句话定义

**模型先推断谁在接触物体、哪些局部会动，再用预测位移引导后续画面。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ReactWorld | ReactWorld: Structuring Interaction Responses for Embodied Video Generation | 本文方法 / 基准的简称 |

## 为什么重要

- 生成机器人视频时，先判断布角会怎样动。在视频生成评测 RBench 上，平均分比 PhysisForcing 高 3.29 分。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-06）。项目页 HTTP 200，未见论文代码链接或 coming soon 字样。
- 同期同组工作见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.06120](https://arxiv.org/abs/2611.06120) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 模型先推断谁在接触物体、哪些局部会动，再用预测位移引导后续画面。 |
| **文内指标** | 公众号转述：在视频生成评测 RBench 上，平均分比 PhysisForcing 高 3.29 分。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见论文代码链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：在视频生成评测 RBench 上，平均分比 PhysisForcing 高 3.29 分。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**生成机器人视频时，先判断布角会怎样动：在视频生成评测 RBench 上，平均分比 PhysisForcing 高 3.29 分。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见论文代码链接或 coming soon 字样。
2. 核心机制：模型先推断谁在接触物体、哪些局部会动，再用预测位移引导后续画面。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-06-20-papers-technology-map](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)

## 参考来源

- [reactworld_arxiv_2611_06120.md](../../sources/papers/reactworld_arxiv_2611_06120.md)
- [wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- [arXiv:2611.06120](https://arxiv.org/abs/2611.06120)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.06120)
- [项目页](https://metaengine-world.github.io/ReactWorld/)
