---
type: overview
tags: [world-models, world-action-models, category-hub, survey]
status: complete
updated: 2026-08-14
summary: "世界模型动作后果专题 · 01 — WAM 直接执行、修正基础 VLA 还是部署前筛选？并链 DreamWAM / FACT / Flex-π / RTCF 邻近坐标。"
related:
  - ./robot-world-models-action-consequence-technology-map.md
  - ./wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
  - ../entities/paper-dreamwam.md
  - ../entities/paper-fact.md
  - ../entities/paper-flex-pi.md
  - ../entities/paper-rtcf.md
  - ../entities/paper-motubrain.md
  - ../entities/paper-rift-wam.md
  - ../entities/paper-wam-realtime-async.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
  - ../../sources/papers/fact_arxiv_2608_10232.md
  - ../../sources/papers/flex_pi_arxiv_2608_10860.md
  - ../../sources/papers/motubrain_arxiv_2604_27792.md
  - ../../sources/papers/rift_wam_arxiv_2608_11521.md
  - ../../sources/papers/wam_realtime_async_arxiv_2608_01880.md
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
| WorldScape Policy 2.0 | [../entities/paper-worldscape-policy-2](../entities/paper-worldscape-policy-2.md) | 同团队下游 WAM 策略：事件记忆走 VLM、视觉记忆走 DiT，多模态提示可控执行 |
| DreamWAM（邻近） | [../entities/paper-dreamwam](../entities/paper-dreamwam.md) | Joint WAM：beyond-RGB 未来表征训练、RGB-only 部署（非本专题原文四篇） |
| FACT（邻近） | [../entities/paper-fact](../entities/paper-fact.md) | 失败感知因果 WAM：失败轨迹教后果；可选 value 筛选候选 |
| Flex-π（邻近） | [../entities/paper-flex-pi](../entities/paper-flex-pi.md) | 多流 Joint WAM + 算力柔性（56 组合；action-only ~60 ms↔full joint ~193 ms） |
| RTCF（邻近） | [../entities/paper-rtcf](../entities/paper-rtcf.md) | **免训练** 冻结 VLA 记忆纠偏；与 DynaWM「可训修正」对照 |
| Motubrain（邻近） | [../entities/paper-motubrain](../entities/paper-motubrain.md) | 生数 Joint WAM；RoboTwin 95.8/96.1（仓占位） |
| WAM 异步部署（邻近） | [../entities/paper-wam-realtime-async](../entities/paper-wam-realtime-async.md) | 同平台六策略：对齐 → blend → train |
| Rift（邻近） | [../entities/paper-rift-wam](../entities/paper-rift-wam.md) | 免视频 rollout：一次 anticipation prefill 写未来 K/V；LIBERO 98.8% / 247.9 ms |

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)
- [DreamWAM](../entities/paper-dreamwam.md)
- [FACT](../entities/paper-fact.md)
- [Flex-π](../entities/paper-flex-pi.md)
- [RTCF](../entities/paper-rtcf.md)
- [Motubrain](../entities/paper-motubrain.md)
- [WAM 实时异步部署](../entities/paper-wam-realtime-async.md)
- [Rift](../entities/paper-rift-wam.md)

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
