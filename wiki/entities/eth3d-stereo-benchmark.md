---
type: entity
tags: [benchmark, stereo-matching, depth-estimation, eth3d, computer-vision]
status: complete
updated: 2026-09-09
related:
  - ./paper-nbs-no-bias-stereo.md
  - ../methods/stereo-matching-foundation-models.md
  - ./middlebury-stereo-benchmark.md
  - ./kitti-stereo-benchmark.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/sites/eth3d_stereo_benchmark.md
summary: "ETH3D Two-View Stereo：高分辨率室内外两视图立体 benchmark；EPE/bad@X 像素指标；NBS 报告该榜 SOTA。"
---

# ETH3D Two-View Stereo Benchmark

**ETH3D** [两视图立体评测](https://www.eth3d.net/low_res_two_view.php) 提供高分辨率 **室内/室外** 校正双目对，以 **EPE**（end-point error）与 **bad@X**（误差超过 X 像素的像素占比）评价稠密视差。[NBS](./paper-nbs-no-bias-stereo.md) 在项目页报告 **ETH3D 全指标第一**（EPE **0.09**，bad@1 **0.16**，bad@4 **0.02**）。

## 一句话定义

**高分辨率、多场景类型的两视图立体「金标准」之一，适合检验亚像素精度与遮挡处理。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ETH3D | ETH 3D Vision Benchmark | 苏黎世联邦理工 3D 视觉基准套件 |
| EPE | End-Point Error | 视差端点误差（像素） |
| bad@X | Bad Pixel @ X px | 误差 > X 的像素比例 |
| OOD | Out-of-Distribution | SimpleProc 等程序生成集与之互补 |

## 核心信息

| 字段 | 内容 |
|------|------|
| **官网** | <https://www.eth3d.net/low_res_two_view.php> |
| **数据** | <https://www.eth3d.net/datasets> |
| **指标** | EPE ↓；bad@1 / bad@4 ↓（像素） |

## 为什么重要

- **与 KITTI 互补：** ETH3D 强调 **高分辨率精细几何**；KITTI 强调 **驾驶场景**。
- **NBS / S²M² / FoundationStereo** 等新方法主战场之一。

## 关联页面

- [NBS](./paper-nbs-no-bias-stereo.md) — ETH3D 榜锚论文
- [立体匹配生态](../methods/stereo-matching-foundation-models.md)
- [Middlebury](./middlebury-stereo-benchmark.md) / [KITTI](./kitti-stereo-benchmark.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页是**模块级**视差精度评测，不落在该闭环 ①–④ 任一层；EPE/bad@X 改善只说明深度栈更准，是否真提升任务表现仍要回到闭环 ③ 层的策略成功率去测

## 参考来源

- [ETH3D 站点归档](../../sources/sites/eth3d_stereo_benchmark.md)
- ETH3D 官网：<https://www.eth3d.net/>

## 推荐继续阅读

- Two-view leaderboard：<https://www.eth3d.net/low_res_two_view.php>
