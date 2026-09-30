---
type: entity
tags:
  - paper
  - vla
  - autonomous-driving
  - geometry
  - map-semantics
status: complete
updated: 2026-09-30
arxiv: "2608.21440"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md
sources:
  - ../../sources/papers/geo_vla_arxiv_2608_21440.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md
summary: "Geo-VLA（arXiv:2608.21440，北科大）：训练期内化道路几何与地图语义，推理无需 HD map；提升多种动作头 VLA 规划，单相机 NAVSIM 最佳档。"
---

# Geo-VLA

**Geo-VLA: Geometry-Aware Vision-Language-Action Planning via Internalization of Map Semantics**（arXiv:[2608.21440](https://arxiv.org/abs/2608.21440)）— **北京科技大学（USTB）**。多模空间 [2026.08.17–08.23 周报](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md) 策展条目；细节以 arXiv 为准。

## 一句话定义

训练期内化道路几何与地图语义，推理无需 HD map；提升多种动作头 VLA 规划，单相机 NAVSIM 最佳档。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习后训练或微调 |
| TTA | Test-Time Augmentation / Adaptation | 测试时增强或适配 |
| SR | Success Rate | 任务成功率 |
| LIBERO | LIBERO Benchmark | 常见操作仿真基准套件 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 北京科技大学（USTB） |
| **评测** | NAVSIM v1 |
| **开源** | 待核实（截至 2026-09-29） |

## 为什么重要

- 纳入 [一周 VLA 趋势（2026.08.17 第一篇）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md) 横切面索引。
- 与 [VLA](../methods/vla.md) 方法页及同周其他 **16/16 独立 canonical 节点** 交叉对照。

## 评测与指标

- **基准：** **NAVSIM v1**（非反应式端到端驾驶规划评测，见 [NAVSIM](./paper-rcl-ref-e25271fa6f028e5611cf-navsim-data-driven-non-reactive-autonomous-vehic.md)）。
- **主结果：** **92.1 PDMS**，论文自报为发表时（2026-08）单相机 VLA 规划器中的最高分；在多种动作生成架构的 VLA 规划器上均带来一致提升（数值摘自 arXiv 摘要，完整表格与基线设定以原文为准）。
- **数据：** 自建 **Geo-QA** 几何问答数据集，用于对比学习 + 指令微调注入道路几何；**推理时不需要高精地图或额外车道信息**。

## 与其他工作对比

| 维度 | Geo-VLA | 对照 |
|------|---------|------|
| 几何信息来源 | 训练时内化地图语义，推理仅单相机 | [OpenDriveVLA](./paper-rcl-ref-b5386c6f934f87f4cec4-opendrivevla-towards-end-to-end-autonomous-drivi.md) 等端到端驾驶 VLA 依赖图像/结构化感知输入 |
| 形态 | 即插即用，增强已有 VLA 规划器 | [AutoVLA](./paper-rcl-2506-13757-autovla-a-vision-language-action-model-for-end-t.md)：完整的端到端驾驶 VLA 模型 |
| 输出统一 | 保留原规划器动作头 | [EMMA（Waymo）](./paper-emma-waymo-e2e.md)：把轨迹/检测/路网统一成自然语言输出 |

## 结论

**Geo-VLA 在本库中作为 arXiv:2608.21440 的 canonical 详情节点；部署与复现前请对照原文 PDF/HTML 与作者发布资源。**

1. **canonical 唯一性** — 全库仅此一页绑定 arXiv:2608.21440。
2. **读法** — 先读公众号策展摘要，再读 arXiv 方法与实验节。
3. **开源** — 待核实（截至 2026-09-29）。
4. **安全/评测类**（若适用）— 勿把任务成功率等同于安全或授权跟随。

## 关联页面

- [VLA](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)
- [一周 VLA 趋势地图（2026.08.17）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md)

## 参考来源

- [geo_vla_arxiv_2608_21440.md](../../sources/papers/geo_vla_arxiv_2608_21440.md)
- [多模空间周报归档](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)
- arXiv：<https://arxiv.org/abs/2608.21440>

## 推荐继续阅读

- [arXiv 摘要页](https://arxiv.org/abs/2608.21440)
