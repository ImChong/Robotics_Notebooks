---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-09-05
summary: "具身世界模型六路线 · 趋势与判断 — 文内五个判断与行业方向所引工作。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 趋势与判断

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **趋势与判断** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| Cosmos 3 | [cosmos-3](../entities/cosmos-3.md) | 统一骨干处理文本/图像/视频/音频/动作的全模态世界基础模型。 |
| NVIDIA Cosmos | [nvidia-cosmos](../entities/nvidia-cosmos.md) | 1.0→2.5→3.0 平台；与 Newton 解析仿真互补。 |
| WorldArena | [paper-sa-2602-08971-worldarena-a-unified-benchmark-for-evaluating-pe](../entities/paper-sa-2602-08971-worldarena-a-unified-benchmark-for-evaluating-pe.md) | 对比视频质量与数据生成/策略评估/规划效用。 |
| RoboWM-Bench | [paper-robowm-bench-action-faithfulness](../entities/paper-robowm-bench-action-faithfulness.md) | 把生成行为还原为机器人动作并在真机执行评测。 |
| DreamDojo | [paper-hrl-stack-35-dreamdojo](../entities/paper-hrl-stack-35-dreamdojo.md) | 第一视角人类视频学日常交互，少量机器人数据恢复可控性。 |
| PlayWorld | [paper-playworld-autonomous-play-data](../entities/paper-playworld-autonomous-play-data.md) | 自主玩耍采集漏抓/滑动/碰撞/形变等失败长尾。 |
| Newton | [newton-physics](../entities/newton-physics.md) | 物理引擎提供几何/接触/约束，与神经 WM 融合。 |
| PIN-WM | [paper-sa-2504-16693-pin-wm-learning-physics-informed-world-models-fo](../entities/paper-sa-2504-16693-pin-wm-learning-physics-informed-world-models-fo.md) | 真实数据识别参数并补充视觉与未建模残差。 |
| Foresight (PI) | [paper-foresight-action-conditioned-failure-monitoring](../entities/paper-foresight-action-conditioned-failure-monitoring.md) | 动作条件表征监测失败风险，服务端侧安全闭环。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
