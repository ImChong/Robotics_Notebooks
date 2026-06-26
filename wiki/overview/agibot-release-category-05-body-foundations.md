---
type: overview
tags: [agibot, motion-control, whole-body-control, category-hub, survey, bfm]
status: complete
updated: 2026-06-26
summary: "智元 2026-06 发布 · 05 身体能力底座 — BFM-2 运动小脑与 AGILE 感控闭环如何补身体层？"
related:
  - ./agibot-june-2026-release-technology-map.md
  - ./agibot-release-category-04-execution-vla.md
  - ./bfm-41-papers-technology-map.md
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/agibot-bfm-2.md
  - ../entities/agibot-agile.md
  - ../concepts/behavior-foundation-model.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# 智元发布分类 05：身体能力底座

> **图谱分类节点**：对应发布会 **两个能力底座**（运动小脑 + 感控闭环）；总地图见 [智元 2026-06 发布技术地图](./agibot-june-2026-release-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 行为基础模型，可复用身体能力接口 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 核心问题

高层规划完整时，**身体层**仍可能失败：**动作间如何过渡？视觉反馈如何进入控制？** 本组两个底座分别回答。

## 本组项目（2 个 · 能力底座）

| # | 项目 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 05 | BFM-2 | **运动小脑** — 动作插值与动态闭环 | [agibot-bfm-2.md](../entities/agibot-bfm-2.md) |
| 06 | AGILE | **感控一体** — 视觉进入控制闭环 | [agibot-agile.md](../entities/agibot-agile.md) |

## 与 GO-2 / BFM 学术线分工

| 层 | 项目 | 分工 |
|----|------|------|
| 语义–动作 | [GO-2](../entities/go-2.md) | 规划到可执行动作序列 |
| 运控底座 | BFM-2 | 动作过渡、肌肉记忆式闭环 |
| 感控底座 | AGILE | 眼–身接口、视觉驱动控制调整 |
| 学术脉络 | [BFM 41 篇](./bfm-41-papers-technology-map.md) | awesome-bfm-papers 问题分类 |

## 关联页面

- [运动小脑 64 篇地图](./humanoid-motion-cerebellum-technology-map.md)
- [语义执行基座](./agibot-release-category-04-execution-vla.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)

## 推荐继续阅读

- [BFM-2 视频（B站）](https://www.bilibili.com/video/BV1ZzGe6oEmk/)
- [AGILE 视频（B站）](https://www.bilibili.com/video/BV1SmGD6xEPS/)
