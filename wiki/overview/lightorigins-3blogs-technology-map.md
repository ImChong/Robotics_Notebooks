---
type: overview
tags: [overview, light-origins, navigation, humanoid, technology-map]
status: complete
updated: 2026-11-04
related:
  - ../queries/lightnav-0-blog-references-deep-read.md
  - ../entities/paper-tolebi.md
  - ../entities/paper-locoformer.md
  - ../entities/paper-humanup-getting-up.md
  - ../entities/lightnav-er.md
  - ../entities/insight-bench.md
  - ../entities/paper-trackvla.md
  - ../entities/paper-rxr.md
  - ../entities/paper-hm3d-ovon.md
  - ../entities/paper-srdf-vln-flywheel.md
  - ../entities/habitatgs.md
  - ../entities/interiorgs.md
  - ../entities/vlnverse.md
  - ../entities/er-point-bench.md
  - ../entities/refspatial.md
  - ../entities/robospatial.md
  - ../entities/where2place.md
  - ../entities/cv-bench-embodied.md
  - ../entities/erqa.md
  - ../entities/embspatial.md
  - ../entities/molmo2-vlm.md
  - ../entities/molmo-er.md
  - ../entities/seed2-0.md
  - ../entities/qwen3-vl.md
  - ../entities/lightbot-0.md
  - ../entities/light-origins.md
  - ../entities/paper-lightnav-0.md
  - ../entities/light-react.md
  - ../entities/paper-light-loco-parkour.md
sources:
  - ../../sources/sites/skild-ai-timeline-audit-2026-10-07.md
  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
  - ../../sources/blogs/lightorigins_lightparkour_2026-08-03.md
original:
  - ../../sources/blogs/lightorigins_lightnav_0_reading_l3bmap.md
  - ../../sources/blogs/lightorigins_lightparkour_reading_l3bmap.md
  - ../../sources/blogs/lightorigins_light_react_reading_l3bmap.md
summary: "亮源新创三篇官方 Tech Blog（LightNav-0 / LightParkour / Light REACT）引用的论文/基准/硬件各有一页，按博客分组可逐条点开。"
---

# 亮源新创三篇 Tech Blog：引用阅读坐标

> **定位：** [LightNav-0](https://www.lightorigins.com/blog/lightnav-0) · [LightParkour](https://www.lightorigins.com/blog/lightparkour) · [Light REACT](https://www.lightorigins.com/blog/light-react)

## 一句话观点

**亮源新创公开的三段范式——预训练（Light-O1）→ 对齐（LightNav-0）→ 部署（Light REACT）——在跑酷线（LightParkour）上还有并行的全身感知运动蒸馏轴；读博客时把「引用论文/基准/硬件」分开看，三类各有一页，不与机构页混在一起。**

## 一句话定义

本页是亮源新创三篇官方 Tech Blog（LightNav-0、LightParkour、Light REACT）的**技术地图与引用索引**：把三条技术线与各自引用的论文、基准、硬件按博客分组，指向库内独立页面。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| R2S2R | Real-to-Sim-to-Real | LightNav 数据引擎核心范式 |
| ER | Embodied Reasoning | 导航后训练第一阶段 |
| ICL | In-Context Learning | Light REACT 全身韧性上下文适应 |
| RVQ | Residual Vector Quantization | LightNav 动作 token 化 |

## LightNav-0 单篇深读

博客 [lightnav-0](https://www.lightorigins.com/en/blog/lightnav-0) 引用项的 **三阶段管线 + 全表索引 + 读榜数字** 见 **[LightNav-0 博客引用深读](../queries/lightnav-0-blog-references-deep-read.md)**（Query 产物，2026-09-22）。

## LocoFormer 归属与日期复核

2026-10-07 核对官方论文与项目页：[LocoFormer](../entities/paper-locoformer.md)署名机构为 **Skild AI**，公司博客日期 **2025-09-24**、arXiv v1 提交 **2025-09-28**。下表的 2026-09 是 Light REACT 博客引用日期，不是 LocoFormer 首发日；本页仅组织亮源博客引用链，不改变被引论文的机构归属。

## 为什么重要

三篇博客共同的方法论是「用仿真/蒸馏把少量真实经验放大」：LightNav-0 放大场景，LightParkour 放大地形，Light REACT 放大受损情形。读者按线索跳转，可避免把三条不同任务线（导航、跑酷、韧性控制）混为一谈。

## 核心原理

（以下均据 `original:` 所列三份博客通读归档。）

| 博客 | 核心机制 | 关键数字（博客自报） |
|------|----------|----------------------|
| LightNav-0 | Real2Sim2Real 数据引擎 → ER 中期训练 → 具身 SFT（Point CoT + 3 个 RVQ 动作 token）→ GRPO 在线 RL | 2,000+ 场景、4,000+ 小时 VLA 经验；Point CoT 平均 SR +8.4；EVT-Bench 干扰跟踪 74.4 → 82.6 |
| LightParkour | 初始动作 → 物理仿真修正接触 → 逐级升高障碍扩展参考轨迹 → 多专家 DAgger → 带记忆深度视觉学生 | 障碍 45 → 75 cm（0.83H）；切换训练端到端 33.0% → 98.0%（仿真） |
| Light REACT | 韧性金字塔四层；6 个 PPO+AMP 教师 → 多教师 DAgger 蒸馏到 64 帧 Transformer → 偏好对齐 RL | 196 种受损情形；对齐后直立占比示意 42% → 76% |

## 工程实践

- LightNav-0 部署只需单目 RGB + 语言，输出图像点与动作 token，不依赖深度、定位或地图；已开源模型与代码。
- LightParkour 真机只运行最终策略（50 Hz，胸前深度相机），训练中模拟延迟与丢帧；定量结果均来自仿真。
- Light REACT 的教师依赖训练期特权的 21 维受损标记，学生仅靠速度指令与本体感知历史判断自身状态。
- 开源状态（据库内归档，截至 2026-09-21）：仅 LightNav-0 列出公开代码与权重。

## 局限

- 三篇均为公司博客，数字为自报，且 LightParkour 的定量结果只在仿真中；75 cm 处部署策略仅 33.4%，与特权教师（98.6%）差距明显。
- LightNav-0 模型超过 4B 后收益不稳定；SFT 数据缺少模型自身犯错后的恢复过程，博客以在线 RL 补充。
- Light REACT 中偏好对齐在 MLP/RNN 上失败的原因，博客给出的是可能性解释而非因果验证；仅比较了三种架构。
- 本页未逐篇核对被引论文原文，被引论文的结论以各自页面为准。

## 完整索引

| 名称 | 博客日期 | 详情 |
|------|----------|------|
| TOLEBI | 2026-09 | [paper-tolebi](../entities/paper-tolebi.md) |
| LocoFormer | 2026-09 | [paper-locoformer](../entities/paper-locoformer.md) |
| HUMANUP | 2026-09 | [paper-humanup-getting-up](../entities/paper-humanup-getting-up.md) |
| LightNav-ER | 2026-09 | [lightnav-er](../entities/lightnav-er.md) |
| INSIGHT-Bench | 2026-09 | [insight-bench](../entities/insight-bench.md) |
| TrackVLA | 2026-09 | [paper-trackvla](../entities/paper-trackvla.md) |
| RxR | 2026-09 | [paper-rxr](../entities/paper-rxr.md) |
| HM3D-OVON | 2026-09 | [paper-hm3d-ovon](../entities/paper-hm3d-ovon.md) |
| SRDF | 2026-09 | [paper-srdf-vln-flywheel](../entities/paper-srdf-vln-flywheel.md) |
| HabitatGS | 2026-09 | [habitatgs](../entities/habitatgs.md) |
| InteriorGS | 2026-09 | [interiorgs](../entities/interiorgs.md) |
| VLNVerse | 2026-09 | [vlnverse](../entities/vlnverse.md) |
| Point-Bench | 2026-09 | [er-point-bench](../entities/er-point-bench.md) |
| RefSpatial | 2026-09 | [refspatial](../entities/refspatial.md) |
| RoboSpatial | 2026-09 | [robospatial](../entities/robospatial.md) |
| Where2Place | 2026-09 | [where2place](../entities/where2place.md) |
| CV-Bench | 2026-09 | [cv-bench-embodied](../entities/cv-bench-embodied.md) |
| ERQA | 2026-09 | [erqa](../entities/erqa.md) |
| EmbSpatial | 2026-09 | [embspatial](../entities/embspatial.md) |
| Molmo2 | 2026-09 | [molmo2-vlm](../entities/molmo2-vlm.md) |
| MolmoER | 2026-09 | [molmo-er](../entities/molmo-er.md) |
| Seed2.0 | 2026-09 | [seed2-0](../entities/seed2-0.md) |
| Qwen3-VL | 2026-09 | [qwen3-vl](../entities/qwen3-vl.md) |
| Lightbot 0 | 2026-08 | [lightbot-0](../entities/lightbot-0.md) |
| BeyondMimic | 2026-09 | [methods/beyondmimic.md](../methods/beyondmimic.md) |
| SONIC | 2026-09 | [methods/sonic-motion-tracking.md](../methods/sonic-motion-tracking.md) |
| HoST | 2026-09 | [entities/paper-host-humanoid-standingup.md](../entities/paper-host-humanoid-standingup.md) |
| PPO | 2026-09 | [methods/ppo.md](../methods/ppo.md) |
| AMP | 2026-09 | [methods/amp-reward.md](../methods/amp-reward.md) |
| Gemini Robotics-ER | 2026-09 | [entities/gemini-robotics.md](../entities/gemini-robotics.md) |
| LightNav-0 | 2026-09 | [entities/paper-lightnav-0.md](../entities/paper-lightnav-0.md) |
| Light REACT | 2026-09 | [entities/light-react.md](../entities/light-react.md) |
| Light-Loco-Parkour | 2026-08 | [entities/paper-light-loco-parkour.md](../entities/paper-light-loco-parkour.md) |
| R2R | 2026-09 | [entities/paper-vln-01-r2r.md](../entities/paper-vln-01-r2r.md) |
| ScaleVLN | 2026-09 | [entities/paper-vln-08-scalevln.md](../entities/paper-vln-08-scalevln.md) |
| VLN-CE | 2026-09 | [entities/paper-vln-02-vln-ce.md](../entities/paper-vln-02-vln-ce.md) |
| DAgger | 2026-08 | [methods/dagger.md](../methods/dagger.md) |

## 关联页面

- [亮源新创（Light Origins）](../entities/light-origins.md)
- [LightNav-0](../entities/paper-lightnav-0.md)
- [Light REACT](../entities/light-react.md)
- [Light-Loco-Parkour](../entities/paper-light-loco-parkour.md)

## 参考来源

- [Skild AI 官方日期与机构核查](../../sources/sites/skild-ai-timeline-audit-2026-10-07.md)

- [lightorigins_light_react_2026-09-09.md](../../sources/blogs/lightorigins_light_react_2026-09-09.md)
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)
- [lightorigins_lightparkour_2026-08-03.md](../../sources/blogs/lightorigins_lightparkour_2026-08-03.md)

## 继续阅读

- [LightNav-0 博客引用深读](../queries/lightnav-0-blog-references-deep-read.md)：三阶段与全部引用项的逐条阅读。
- 原文通读归档：[LightNav-0](../../sources/blogs/lightorigins_lightnav_0_reading_l3bmap.md) · [LightParkour](../../sources/blogs/lightorigins_lightparkour_reading_l3bmap.md) · [Light REACT](../../sources/blogs/lightorigins_light_react_reading_l3bmap.md)
