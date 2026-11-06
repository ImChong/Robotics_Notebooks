---
type: entity
project_id: messynav
tags:
  - paper
  - navigation
  - evaluation
status: complete
updated: 2026-11-06
arxiv: "2611.07417"
project: https://messynav.github.io/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-06-20-papers-technology-map.md
sources:
  - ../../sources/papers/messynav_arxiv_2611_07417.md
  - ../../sources/sites/messynav.md
  - ../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md
summary: "MessyNav（arXiv:2611.07417）：空推车可以挪，放着笔记本的推车却该绕开。"
---

# MessyNav（arXiv:2611.07417）

**MessyNav**（*MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments*，[arXiv:2611.07417](https://arxiv.org/abs/2611.07417)，[项目页](https://messynav.github.io/)）来自 [具身智能小站 2026-11-06 盘点](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)，属「导航与闭环评测」组。

## 一句话定义

**空推车可以挪，放着笔记本的推车却该绕开。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MessyNav | MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments | 本文方法 / 基准的简称 |

## 为什么重要

- 路被小车堵住，机器人该不该推。MessyNav 先判断障碍物能否安全移动，再选动作。
- 开源结论：**待发布**（步骤 2.5，2026-11-06）。项目页 HTTP 200，标注 Video and code coming soon。
- 同期同组工作见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.07417](https://arxiv.org/abs/2611.07417) |
| **开源** | **待发布** |
| **要点** | 空推车可以挪，放着笔记本的推车却该绕开。 |
| **文内指标** | 公众号转述：15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，标注 Video and code coming soon。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**路被小车堵住，机器人该不该推：15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。**

1. 开源边界：**待发布** — 项目页 HTTP 200，标注 Video and code coming soon。
2. 核心机制：空推车可以挪，放着笔记本的推车却该绕开。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-06-20-papers-technology-map](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)

## 参考来源

- [messynav_arxiv_2611_07417.md](../../sources/papers/messynav_arxiv_2611_07417.md)
- [wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- [arXiv:2611.07417](https://arxiv.org/abs/2611.07417)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.07417)
- [项目页](https://messynav.github.io/)
