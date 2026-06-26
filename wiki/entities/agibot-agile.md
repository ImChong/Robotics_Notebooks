---
type: entity
tags: [agibot, perception, whole-body-control, visuomotor]
status: complete
updated: 2026-06-26
related:
  - ../overview/agibot-june-2026-release-technology-map.md
  - ../overview/agibot-release-category-05-body-foundations.md
  - ./agibot-bfm-2.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
summary: "AGILE 是智元感控一体能力底座：让视觉进入控制闭环，使灵犀 X2 等本体据视觉反馈调整身体与末端动作；与 BFM-2 运控小脑互补。"
---

# AGILE（智元感控一体）

**AGILE** 是智元在 [2026-06 发布地图](../overview/agibot-june-2026-release-technology-map.md) 中推出的 **感控一体能力底座**（公开视频：[B站 BV1SmGD6xEPS](https://www.bilibili.com/video/BV1SmGD6xEPS/)）。视频标题强调 **「视觉觉醒」**：让 [灵犀 X2](./agibot-lingxi-x1.md) 等本体实现 **感知与控制融合**。

## 一句话定义

**眼–身闭环接口**——看见环境之后，控制策略能否及时随视觉反馈调整全身与末端动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| IL | Imitation Learning | 从示范轨迹学习策略 |

## 为什么重要

- **能力底座位：** 与 [BFM-2](./agibot-bfm-2.md) 并列的 **两个能力底座之二**。
- **分工：** BFM-2 更偏 **运动小脑 / 动作过渡**；AGILE 更偏 **视觉进入控制过程**。
- **连续执行：** 真实任务中环境变化要求 **感知–控制同环**，否则高层 VLA 规划易被底层偏差打断。

## 命名辨析

- 本页 **AGILE** 指智元 **感控一体产品/模型**；仓库内另有论文实体含 “agile” 字样（如四足/人形 loco 论文），**勿合并节点**。

## 信息边界

- 截至 ingest 时，公开材料主要为 **视频标题与简介**；机制细节须以后续技术报告为准。

## 关联页面

- [身体能力底座分类 hub](../overview/agibot-release-category-05-body-foundations.md)
- [全身控制](../concepts/whole-body-control.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)

## 推荐继续阅读

- [AGILE 公开视频](https://www.bilibili.com/video/BV1SmGD6xEPS/)
