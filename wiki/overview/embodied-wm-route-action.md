---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
summary: "具身世界模型六路线 · 行动主导型 — 部署时动作生成与未来建模显式耦合。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 行动主导型

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **行动主导型** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| Unified World Models (UWM) | [paper-shenlan-wm-08-uwm](../entities/paper-shenlan-wm-08-uwm.md) | 未来观测扩散与动作扩散同一 Transformer，未来目标改善动作。 |
| Cosmos Policy | [paper-shenlan-wm-11-cosmos-policy](../entities/paper-shenlan-wm-11-cosmos-policy.md) | 动作块/未来视觉/本体/价值同一潜在空间，可想象多未来再排序。 |
| DreamZero | [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) | 同一生成过程输出未来视频与动作，真实观测修正下一轮。 |
| Riemann-1.0 | [paper-riemann-1-causal-action-video-wam](../entities/paper-riemann-1-causal-action-video-wam.md) | 统一因果序列：历史观测先生成动作，再条件预测视觉后果；真机可直接执行动作。 |
| World Tokens | [paper-world-tokens-inference-trimmed-wam](../entities/paper-world-tokens-inference-trimmed-wam.md) | 训练期世界监督、推理期裁剪生成分支的 WAM 趋势代表。 |
| FLEX-π | [paper-flex-pi](../entities/paper-flex-pi.md) | RGB/点图/语义共同塑造未来表征的多流 Joint WAM。 |
| MobileWAM | [paper-mobilewam-mobile-manipulation-wam](../entities/paper-mobilewam-mobile-manipulation-wam.md) | 从机械臂扩展到移动操作的 WAM。 |
| MotionWAM | [paper-motionwam-humanoid-loco-manipulation-wam](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) | 实时人形 loco-manipulation：Video DiT 隐状态条件 Motion DiT。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
