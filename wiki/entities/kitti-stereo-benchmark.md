---
type: entity
tags: [benchmark, stereo-matching, kitti, autonomous-driving, computer-vision]
status: complete
updated: 2026-09-09
related:
  - ../methods/stereo-matching-foundation-models.md
  - ./eth3d-stereo-benchmark.md
  - ../concepts/state-estimation.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/sites/kitti_stereo_benchmark.md
summary: "KITTI Stereo 2012/2015：自动驾驶场景最常用双目立体 benchmark；D1-all、EPE 等指标。"
---

# KITTI Stereo Benchmark

**KITTI** 立体评测（[2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) / [2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)）基于车载双目序列，是 **自动驾驶** 场景下最常被引用的立体匹配 benchmark，主指标包括 **D1-all**（视差误差 >3px 且相对误差 >5% 的像素比例）与 **EPE**。

## 一句话定义

**驾驶场景立体匹配的默认考场——强调远距离、弱纹理路面与真实车速运动。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| KITTI | Karlsruhe Institute of Technology and Toyota | 数据集名 |
| D1 | Disparity Error > 3px | 坏点定义之一 |
| EPE | End-Point Error | 视差端点误差 |
| SF | Scene Flow | 2015 评测含光流联合任务 |

## 核心信息

| 字段 | 内容 |
|------|------|
| **KITTI 2012** | <https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo> |
| **KITTI 2015** | <https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo> |

## 为什么重要

- **机器人/车规感知：** 室外移动平台深度栈常先在 KITTI 上对标，再转真机标定。
- **与 ETH3D 不可横比：** 分辨率、场景、指标定义均不同。

## 关联页面

- [立体匹配生态](../methods/stereo-matching-foundation-models.md)
- [ETH3D](./eth3d-stereo-benchmark.md)
- [State Estimation](../concepts/state-estimation.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页是**模块级**视差精度评测，不落在该闭环 ①–④ 任一层；D1-all 降低只说明深度栈更准，下游导航/操作是否受益仍要回到闭环 ③ 层的策略成功率去测

## 参考来源

- [KITTI 站点归档](../../sources/sites/kitti_stereo_benchmark.md)

## 推荐继续阅读

- KITTI Stereo 2015 leaderboard：<https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo>
