---
type: overview
tags: [world-models, world-action-models, category-hub, survey]
status: complete
updated: 2026-07-11
summary: "世界模型动作后果专题 · 01（4 篇）— WAM 直接执行、修正基础 VLA 还是部署前筛选？"
related:
  - ./robot-world-models-action-consequence-technology-map.md
  - ./wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
---

# 世界模型动作后果分类 01：WAM 动作后果预测

> **图谱分类节点**：对应 [具身智能研究室 · 世界模型动作后果专题](https://mp.weixin.qq.com/s/a5ZDDv70CLDfY98mfviWuA) 的 **01 WAM 动作后果预测** 段；总地图见 [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)。

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
| DSWAM | [../entities/paper-dswam-dual-system-wam](../entities/paper-dswam-dual-system-wam.md) | 双系统 WAM 执行器 + 可选 VLM 规划器；视频协同训练、推理直出动作 |
| DynaWM | [../entities/paper-dynawm-vla-online-correction](../entities/paper-dynawm-vla-online-correction.md) | 冻结 VLA + Mamba/V-JEPA 条件流匹配在线重生成动作块 |
| DreamSteer | [../entities/paper-dreamsteer-vla-deployment-steering](../entities/paper-dreamsteer-vla-deployment-steering.md) | 潜变量 WM 预演候选动作 + 语言价值模型排序 |
| Worldscape-MoE | [../entities/paper-worldscape-moe-heterogeneous-action](../entities/paper-worldscape-moe-heterogeneous-action.md) | 共享/专属 MoE 统一相机、关节、手部异构动作接口 |

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
