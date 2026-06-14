---
type: overview
tags: [loco-manipulation, whole-body-control, category-hub, survey, command-space]
status: complete
updated: 2026-06-14
summary: "Loco-Manip 8 篇周报 · 03 命令空间与控制器（2 篇）— 多源数据如何落到解耦命令（VAIC）与多模态统一 WBC（M3imic）？"
related:
  - ./loco-manip-8-papers-technology-map.md
  - ./loco-manip-category-02-synthetic-data.md
  - ./loco-manip-category-04-contact-teleop.md
  - ./bfm-41-papers-technology-map.md
  - ../entities/paper-loco-manip-05-vaic.md
  - ../entities/paper-loco-manip-06-m3imic.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md
  - ../../sources/papers/loco_manip_8_papers_catalog.md
---

# Loco-Manip 分类 03：命令空间与控制器

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 8 篇周报](https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ) 的 **03 命令空间与控制器** 分组；总地图见 [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| BFM | Behavior Foundation Model | 可复用、可适配的全身行为基座 |

## 核心问题

**数据入口变多之后，命令接口是否成为瓶颈？** 参考轨迹过密会绑死策略；命令过粗则无法表达物体交互阶段。须设计 **解耦命令** 或 **多模态 latent 命令空间**，再由统一 WBC 执行。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 05 | VAIC | [paper-loco-manip-05-vaic.md](../entities/paper-loco-manip-05-vaic.md) | [source](../../sources/papers/loco_manip_survey_05_vaic.md) |
| 06 | M3imic | [paper-loco-manip-06-m3imic.md](../entities/paper-loco-manip-06-m3imic.md) | [source](../../sources/papers/loco_manip_survey_06_m3imic.md) |

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md)
- [BFM 技术地图](./bfm-41-papers-technology-map.md)
- [触觉与跨本体遥操作](./loco-manip-category-04-contact-teleop.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_8_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)
- [loco_manip_8_papers_catalog.md](../../sources/papers/loco_manip_8_papers_catalog.md)

## 推荐继续阅读

- [M3imic 代码](https://github.com/Renforce-Dynamics/MultiModalWBC)
