---
type: overview
tags: [world-models, world-action-models, category-hub, survey]
status: complete
updated: 2026-07-11
summary: "世界模型动作后果专题 · 02（4 篇）— 触觉、电流与形变如何进入状态转移？"
related:
  - ./robot-world-models-action-consequence-technology-map.md
  - ./wm-action-consequence-category-03-geometry-4d.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
---

# 世界模型动作后果分类 02：接触状态建模

> **图谱分类节点**：对应 [具身智能研究室 · 世界模型动作后果专题](https://mp.weixin.qq.com/s/a5ZDDv70CLDfY98mfviWuA) 的 **02 接触状态建模** 段；总地图见 [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来观测与动作生成的具身策略 |
| VLA | Vision-Language-Action | 常被修正、筛选或后训练的上层策略 |
| WM | World Model | 预测动作后果的潜变量或视频模型 |
| MoE | Mixture of Experts | 异构动作模态的专家混合架构 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| VT-WAM | [../entities/paper-vt-wam-visuotactile-contact-rich](../entities/paper-vt-wam-visuotactile-contact-rich.md) | 视觉-触觉形变与动作联合流匹配 + 接触门控注意力 |
| TACO | [../entities/paper-taco-tactile-wm-vla-posttrain](../entities/paper-taco-tactile-wm-vla-posttrain.md) | 识别-想象-标注闭环，把失败片段转为 VLA 后训练数据 |
| Current as Touch | [../entities/paper-current-as-touch-proprioceptive-contact](../entities/paper-current-as-touch-proprioceptive-contact.md) | 电机电流/关节状态预测柔顺参考位置 |
| Deform360 | [../entities/paper-deform360-deformable-visuotactile-dataset](../entities/paper-deform360-deformable-visuotactile-dataset.md) | 198 类可变形物体大规模视触觉数据集 |

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
