---
type: entity
tags:
  - paper
  - world-action-models
  - tactile
  - contact-rich
  - flow-matching
  - manipulation
status: complete
updated: 2026-07-11
arxiv: "2607.02503"
related:
  - ../overview/wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-taco-tactile-wm-vla-posttrain.md
  - ../entities/paper-current-as-touch-proprioceptive-contact.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "VT-WAM（arXiv:2607.02503）：统一流匹配联合预测未来视觉、触觉形变与动作；非对称 MoT 注意力 + 接触门控 AVTAG；六类真机接触任务平均成功率 71.67%（+26.67pp vs Fast-WAM）。"
---

# VT-WAM（Visual-Tactile World Action Model）

**VT-WAM**（arXiv:2607.02503，[项目页](https://vt-wam.github.io/)）——见策展导读与一手论文。

## 一句话定义

**联合学习视觉、触觉形变动力学与动作，接触阶段提高对触觉证据的依赖**——而非把触觉当静态附加输入。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 视觉-触觉-动作联合模型 |
| MoT | Mixture-of-Transformers | 非对称跨模态注意力 |
| AVTAG | Action-Visual-Tactile Attention Guidance | 接触门控触觉注意力监督 |

## 为什么重要

- 接触丰富任务中触觉 **时序稀疏但因果关键**；须预测形变如何随动作演化。
- 平均成功率 71.67%，较 Fast-WAM +26.67pp、OmniVTLA +35.84pp。
- 与 [TACO](../entities/paper-taco-tactile-wm-vla-posttrain.md)、[Current as Touch](../entities/paper-current-as-touch-proprioceptive-contact.md) 构成 **触觉/本体接触** 三角。

## 核心结构

| 模块 | 作用 |
|------|------|
| 视觉专家 | 首帧场景锚点 + 未来视觉预测 |
| 触觉专家 | 完整触觉序列与形变预测 |
| 非对称 MoT | 桥接视觉锚点与触觉时序 |
| AVTAG | 接触建立后强化动作对触觉的注意力 |

## 实验要点（策展口径）

六类真机接触任务平均 71.67%。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-02-contact-modeling](../overview/wm-action-consequence-category-02-contact-modeling.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.02503](https://arxiv.org/abs/2607.02503) — 一手论文
- [https://vt-wam.github.io/](https://vt-wam.github.io/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.02503](https://arxiv.org/abs/2607.02503)
