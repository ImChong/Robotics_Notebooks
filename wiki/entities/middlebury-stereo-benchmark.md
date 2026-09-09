---
type: entity
tags: [benchmark, stereo-matching, middlebury, computer-vision]
status: complete
updated: 2026-09-09
related:
  - ../methods/stereo-matching-foundation-models.md
  - ./eth3d-stereo-benchmark.md
  - ./paper-nbs-no-bias-stereo.md
sources:
  - ../../sources/sites/middlebury_stereo_benchmark.md
summary: "Middlebury Stereo V3：经典立体匹配评测；S²M²、FoundationStereo、Selective-Stereo 等常在此对比。"
---

# Middlebury Stereo Evaluation

**Middlebury Stereo** [V3 评测](https://vision.middlebury.edu/stereo/eval3/) 是立体匹配领域 **历史最久** 的公开基准之一，实验场景受控、GT 精细，广泛用于比较 **亚像素视差** 算法。[S²M²](https://github.com/junhong-3dv/s2m2)、**FoundationStereo**、**Selective-Stereo** 等论文常报 Middlebury 成绩。

## 一句话定义

**立体匹配领域的「经典考场」——场景多样但规模小于现代驾驶/合成大数据基准。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MVS | Multi-View Stereo | Middlebury 亦含 MVS 评测（与 two-view stereo 不同轨） |
| RMSE | Root Mean Square Error | 部分 Middlebury 指标 |
| bad | Bad Pixel Percentage | 误差超阈值像素占比 |

## 核心信息

| 字段 | 内容 |
|------|------|
| **官网** | <https://vision.middlebury.edu/stereo/> |
| **V3 评测** | <https://vision.middlebury.edu/stereo/eval3/> |

## 关联页面

- [立体匹配生态](../methods/stereo-matching-foundation-models.md)
- [ETH3D](./eth3d-stereo-benchmark.md) / [KITTI](./kitti-stereo-benchmark.md)

## 参考来源

- [Middlebury 站点归档](../../sources/sites/middlebury_stereo_benchmark.md)

## 推荐继续阅读

- Middlebury V3：<https://vision.middlebury.edu/stereo/eval3/>
