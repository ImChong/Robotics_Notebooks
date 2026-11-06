---
type: entity
project_id: spatial-umi-dream
tags:
  - paper
  - world-model
  - video-generation
status: complete
updated: 2026-11-06
arxiv: "2611.07229"
project: https://weiduoyuan.com/spatial-umi-dream/
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../tasks/teleoperation.md
  - ../overview/embodied-station-2026-11-06-20-papers-technology-map.md
sources:
  - ../../sources/papers/spatial_umi_dream_arxiv_2611_07229.md
  - ../../sources/sites/spatial-umi-dream.md
  - ../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md
summary: "Spatial-UMI-Dream（arXiv:2611.07229）：夹爪靠近目标时，画面变化既来自物体，也来自相机自身。"
---

# Spatial-UMI-Dream（arXiv:2611.07229）

**Spatial-UMI-Dream**（*Spatial-UMI-Dream: Scaling World Action Models with Metric Geometry for UMI Manipulation*，[arXiv:2611.07229](https://arxiv.org/abs/2611.07229)，[项目页](https://weiduoyuan.com/spatial-umi-dream/)）来自 [具身智能小站 2026-11-06 盘点](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)，属「交互式世界模型」组。

## 一句话定义

**夹爪靠近目标时，画面变化既来自物体，也来自相机自身。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Spatial-UMI-Dream | Spatial-UMI-Dream: Scaling World Action Models with Metric Geometry for UMI Manipulation | 本文方法 / 基准的简称 |
| UMI | Universal Manipulation Interface | 手持通用操作采集接口 |

## 为什么重要

- 给手腕相机补上深度，双手更容易对准。模型用双目深度学距离，同时预测图像、深度和动作。
- 开源结论：**部分开源**（步骤 2.5，2026-11-06）。公众号未给代码链接；项目页链接的 Yewandou7/spatial_umi_dream 经 git ls-remote 可达，是否即论文完整代码未核对。
- 同期同组工作见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.07229](https://arxiv.org/abs/2611.07229) |
| **开源** | **部分开源** |
| **要点** | 夹爪靠近目标时，画面变化既来自物体，也来自相机自身。 |
| **文内指标** | 公众号转述：六项双臂真机任务各测 20 次，平均成功率为 85%。 |

## 源码运行时序图

**部分开源** — 公众号未给代码链接；项目页链接的 Yewandou7/spatial_umi_dream 经 git ls-remote 可达，是否即论文完整代码未核对。 待可运行代码公开后补图。

## 实验与评测

- 公众号转述：六项双臂真机任务各测 20 次，平均成功率为 85%。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**给手腕相机补上深度，双手更容易对准：六项双臂真机任务各测 20 次，平均成功率为 85%。**

1. 开源边界：**部分开源** — 公众号未给代码链接；项目页链接的 Yewandou7/spatial_umi_dream 经 git ls-remote 可达，是否即论文完整代码未核对。
2. 核心机制：夹爪靠近目标时，画面变化既来自物体，也来自相机自身。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-data-collection-four-layers-taxonomy](../concepts/embodied-data-collection-four-layers-taxonomy.md)
- [teleoperation](../tasks/teleoperation.md)
- [embodied-station-2026-11-06-20-papers-technology-map](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)

## 参考来源

- [spatial_umi_dream_arxiv_2611_07229.md](../../sources/papers/spatial_umi_dream_arxiv_2611_07229.md)
- [wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- [arXiv:2611.07229](https://arxiv.org/abs/2611.07229)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.07229)
- [项目页](https://weiduoyuan.com/spatial-umi-dream/)
