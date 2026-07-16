---
type: overview
tags: [loco-manipulation, contact-rich, category-hub, survey, vla, world-models]
status: complete
updated: 2026-07-06
summary: "Loco-Manip 接触专题 · 05 VLA/WM（7 篇）— 上层模型能否调用带接触结构的全身动作接口？"
related:
  - ./loco-manip-contact-technology-map.md
  - ./loco-manip-contact-category-04-post-contact-stability.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ./loco-manip-161-category-09-vla-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Loco-Manip 接触分类 05：VLA 与世界模型调用

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) 的 **05 VLA/WM 调用** 段；总地图见 [接触五段链路技术地图](./loco-manip-contact-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 预测动态与后果；MotionWAM 为 WAM 路线 |
| WAM | World Action Model | 世界动作模型 + 全身 motion token |
| WBC | Whole-Body Control | WOLF-VLA 用全身最优控制生成训练数据 |

## 核心问题

VLA/WM 若只输出目标点或粗动作，**接触问题仍被丢回底层**。关键在于上层能否调用 **带接触结构的全身动作接口**；若底层无力/接触接口，预测再合理也难上真机。

## 本组工作（7 篇）

| 工作 | Wiki 实体（复用） | 文内角色 |
|------|-------------------|----------|
| OpenHLM | [paper-loco-manip-161-154-openhlm](../entities/paper-loco-manip-161-154-openhlm.md) | 全身原生人形 VLA |
| WholeBodyVLA | [paper-hrl-stack-30-wholebodyvla](../entities/paper-hrl-stack-30-wholebodyvla.md) | 移动+操作统一潜变量 |
| ROVE | [paper-rove-humanoid-vla-intervention](../entities/paper-rove-humanoid-vla-intervention.md) | 人工接管数据的后训练去噪 |
| MotionWAM | [paper-motionwam-humanoid-loco-manipulation-wam](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) | WAM + motion token 实时 loco-manip |
| ABot-M0.5 | [paper-abot-m05-mobile-manipulation-wam](../entities/paper-abot-m05-mobile-manipulation-wam.md) | 移动操作 WAM：latent action + Dream Forcing |
| HAIC | [haic](../methods/haic.md) | 动态占据、碰撞边界与接触可供性 |
| WOLF-VLA | [paper-wolf-vla](../entities/paper-wolf-vla.md) | 全身最优控制生成动态一致 VLA 数据 |

## 策展判断

- **OpenHLM / WholeBodyVLA：** 避免上肢与下肢 **割裂** 的全身任务模型。
- **ROVE：** 接管数据含犹豫与错误，直接模仿会 **污染** 策略。
- **MotionWAM / ABot-M0.5 / HAIC / WOLF-VLA：** 把问题推向 **世界模型、动力学感知与最优控制数据工厂**。
- **共同结论：** VLA/WM 负责理解、预测与调度，但最终须落到 **带接触结构的身体接口**。

## 关联页面

- [VLA](../methods/vla.md)
- [世界动作模型](../concepts/world-action-models.md)
- [161 篇 · 09 VLA/WM](./loco-manip-161-category-09-vla-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [机器人世界模型训练闭环](./robot-world-models-training-loop-taxonomy.md)
- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
