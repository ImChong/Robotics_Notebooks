---
type: overview
tags: [overview, survey, vla, vln, world-model, technology-map, duomo-space]
status: complete
updated: 2026-09-19
related:
  - ../entities/paper-mamba-smolvla-expert.md
  - ../entities/paper-vane.md
  - ../entities/paper-tdhd-surgical-dual-arm.md
  - ../entities/paper-hermite-curves-vla-trajectory-priors.md
  - ../entities/paper-cross-view-action-consistency-vla.md
  - ../entities/paper-vla-action-post-training-depth-decodability.md
  - ../entities/paper-hymes-hybrid-memory-manipulation.md
  - ../entities/paper-onevomemory.md
  - ../entities/paper-wa-specdec.md
  - ../entities/paper-depth-wise-probing-driving-vla.md
  - ../entities/paper-tempo.md
  - ../entities/paper-recoverfly-aerial-vln.md
  - ../entities/paper-wnm-3d-vln.md
  - ../entities/paper-anycam-vla.md
  - ../entities/paper-wam-diff2.md
  - ../entities/paper-slim-05b.md
  - ../entities/paper-world-tokens-inference-trimmed-wam.md
  - ../entities/paper-omega-0.md
  - ../entities/paper-jepa-wam.md
  - ../entities/paper-activefly-bench.md
  - ../entities/paper-cmu-drive-v2v-vla.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "多模空间 2026-09-17 策展：2026.08.10–08.16 一周 21 篇 VLA/VLN/WAM 论文按架构、记忆、训练、空间感知、世界模型与评测六轴索引。"
---

# 一周 VLA 研究趋势（2026.08.10–08.16 · 第一篇）

> **本页定位**：为 [多模空间公众号盘点](https://mp.weixin.qq.com/s/uKFzE3jyplG7EwbsFG52kw) 提供横切面索引；**21/21 各有独立 `paper-*` 详情节点**，本页不替代单篇深读。

## 一句话观点

**本周 VLA 主线：在不大改 VLM 骨干的前提下，用更轻的 action 结构、更可靠的部署适配、更明确的未来信号接口，以及可闭环的评测基准，把「能跑」推进到「能长期跑、能换相机跑、能协作跑」。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注在线适配 |
| V2V | Vehicle-to-Vehicle | 车–车协同通信 |

## 节点索引（21/21）

### 架构模块

- [Mamba SmolVLA Expert](../entities/paper-mamba-smolvla-expert.md) — arXiv:2608.21407
- [VANE](../entities/paper-vane.md) — arXiv:2608.09448
- [TDHD](../entities/paper-tdhd-surgical-dual-arm.md) — arXiv:2608.09125
- [Hermite Curves VLA](../entities/paper-hermite-curves-vla-trajectory-priors.md) — arXiv:2608.01265
- [Cross-View Action Consistency](../entities/paper-cross-view-action-consistency-vla.md) — arXiv:2608.06965

### 分析诊断 · 记忆 · 异常

- [VLA Depth Decodability](../entities/paper-vla-action-post-training-depth-decodability.md) — arXiv:2608.08904
- [HyMeS](../entities/paper-hymes-hybrid-memory-manipulation.md) — arXiv:2608.09410
- [OnEvoMemory](../entities/paper-onevomemory.md) — arXiv:2608.08749
- [WA-SpecDec](../entities/paper-wa-specdec.md) — arXiv:2608.08725

### 性能 · 训练范式

- [Depth-Wise Probing Driving VLA](../entities/paper-depth-wise-probing-driving-vla.md) — arXiv:2608.07361
- [TEMPO](../entities/paper-tempo.md) — arXiv:2608.07314（复用）
- [RecoverFly](../entities/paper-recoverfly-aerial-vln.md) — arXiv:2608.09467

### 空间感知

- [WNM-3D](../entities/paper-wnm-3d-vln.md) — arXiv:2608.07267
- [AnyCamVLA](../entities/paper-anycam-vla.md) — arXiv:2603.05868

### 世界模型

- [WAM-Diff2](../entities/paper-wam-diff2.md) — arXiv:2608.01035
- [SLIM-0.5B](../entities/paper-slim-05b.md) — arXiv:2608.09771（复用）
- [World Tokens](../entities/paper-world-tokens-inference-trimmed-wam.md) — arXiv:2608.09730（升级）
- [ω-0](../entities/paper-omega-0.md) — arXiv:2608.06375（复用）
- [JEPA-WAM](../entities/paper-jepa-wam.md) — arXiv:2608.09381

### 评测 · 多智能体

- [ActiveFly-Bench](../entities/paper-activefly-bench.md) — arXiv:2607.10180
- [CMU-Drive / V2V-VLA](../entities/paper-cmu-drive-v2v-vla.md) — arXiv:2608.07621

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [World Action Models](../concepts/world-action-models.md)
- [VLA 部署 12 篇地图](./vla-deploy-12-papers-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [具身世界模型六路线地图](./embodied-wm-six-routes-technology-map.md)
