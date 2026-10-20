---
type: entity
tags:
  - dataset
  - tactile
  - audio
  - multisensory
  - real-object
status: draft
updated: 2026-10-20
summary: "100 个真实家居物体的多感官测量：三维网格、高清旋转视频、撞击声与触觉读数。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ObjectFolder Real

## 一句话定义

**ObjectFolder Real**：真实物体版的多感官数据集，配套 ObjectFolder Benchmark 的 10 项任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ObjectFolder | 物体文件（数据集名） | 多感官物体数据 |
| Benchmark | ObjectFolder Benchmark | 10 项任务基准 |
| CVPR | Conference on Computer Vision and Pattern Recognition | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Stanford University |
| 时间 | 2023-06（CVPR 2023） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 100 个真实家居物体 |
| 视觉 | 三种分辨率三维网格；灯箱旋转高清视频 |
| 声学与触觉 | 30–50 个位置各 6 秒撞击声，附力曲线与视频；触觉读数 |

## 为什么重要

- 与 ObjectFolder 2.0 的仿真数据对应，用于检验仿真学到的多感官表征能否用于真实物体。

## 核心原理

对每个物体在多个位置敲击并同步录制声音、力与视频。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | objectfolder.stanford.edu 下载页（文章给出链接） |

## 局限与风险

- 仅 100 个物体，规模小于仿真版（文章数字）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://objectfolder.stanford.edu/objectfolder-real-download)
