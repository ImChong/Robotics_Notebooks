---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
summary: "具身世界模型六路线 · 模型构建型 — 学习并验证世界预测器本身；输出侧重预测精度、物理一致性与 rollout。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 模型构建型

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **模型构建型** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| ContactNets | [paper-contactnets-contact-dynamics](../entities/paper-contactnets-contact-dynamics.md) | 结构化状态空间学习接触几何与物理约束的动力学；以预测精度与穿透检验为终点。 |
| GAIA-1 | [paper-gaia1](../entities/paper-gaia1.md) | 驾驶视频+文本+自车动作统一编码的动作条件视觉未来生成。 |
| Cosmos Predict | [paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi](../entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md) | NVIDIA 从大规模视频与 Physical AI 数据学习时空先验，可后训练到机器人/自动驾驶。 |
| Qwen-RobotWorld | [paper-sa-2606-17030-qwen-robotworld-unifying-embodied-world-modeling](../entities/paper-sa-2606-17030-qwen-robotworld-unifying-embodied-world-modeling.md) | 自然语言作统一动作接口，跨操作/驾驶/导航与人到机器人迁移预测视觉未来。 |
| Genie | [paper-sa-2402-15391-genie-generative-interactive-environments](../entities/paper-sa-2402-15391-genie-generative-interactive-environments.md) | 从无动作标签视频发现可交互潜在控制的可探索环境。 |
| Matrix-Game 3.5 | [paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact](../entities/paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md) | 720p 实时流式交互世界与分钟级场景记忆；策展口径对应 Matrix-Game 3.x 线。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
