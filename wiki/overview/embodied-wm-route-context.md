---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
summary: "具身世界模型六路线 · 上下文主导型 — 持续维护可查询世界状态与记忆（World Proxy）。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 上下文主导型

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **上下文主导型** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| Hydra | [paper-hydra-0](../entities/paper-hydra-0.md) | 实时分层三维场景图维护对象/房间/建筑关系。 |
| ConceptGraphs | [paper-conceptgraphs-open-vocabulary-3d-scene](../entities/paper-conceptgraphs-open-vocabulary-3d-scene.md) | VLM 语义融入三维对象图，支持开放词汇空间查询。 |
| HoloAgent-0 | [holoagent](../entities/holoagent.md) | 空间与时间记忆连接技能系统与失败恢复。 |
| SayPlan | [paper-sayplan-llm-scene-graph-planning](../entities/paper-sayplan-llm-scene-graph-planning.md) | 从大型场景图检索任务子图并用符号约束检查计划。 |
| RoboMemory | [paper-robomemory-multi-type-embodied-memory](../entities/paper-robomemory-multi-type-embodied-memory.md) | 并行维护时间/空间/语义/任务经历供规划与执行调用。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
