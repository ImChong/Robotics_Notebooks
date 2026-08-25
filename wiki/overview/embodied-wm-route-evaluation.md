---
type: overview
tags: [world-models, category-hub, survey, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
summary: "具身世界模型六路线 · 评估主导型 — 外部策略在学习世界中考试，减少真机筛选。"
related:
  - ./embodied-wm-six-routes-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
---

# 具身世界模型六路线 · 评估主导型

> **图谱分类节点**：对应 [六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **评估主导型** 段；总地图见 [embodied-wm-six-routes-technology-map](./embodied-wm-six-routes-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测未来状态/观测的内部模型 |
| WAM | World Action Model | 联合未来与动作的具身策略 |
| MPC | Model Predictive Control | 模型内滚动搜索动作 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| WorldGym | [paper-shenlan-wm-15-worldgym](../entities/paper-shenlan-wm-15-worldgym.md) | 外部策略在 WM 闭环 rollout，用相对排序减少真机筛选。 |
| Veo World Simulator | [paper-veo-world-simulator-policy-testing](../entities/paper-veo-world-simulator-policy-testing.md) | 视频基础模型改造为机器人测试场：OOD 与安全红队。 |
| GE-Sim 2.0 | [ge-sim-2](../entities/ge-sim-2.md) | 动作条件多视角视频+本体反馈的闭环策略评测模拟器。 |
| WorldEval | [paper-sa-2505-19017-worldeval-world-model-as-real-world-robot-polici](../entities/paper-sa-2505-19017-worldeval-world-model-as-real-world-robot-polici.md) | 轻量策略条件世界模型评估。 |
| GigaWorld-1 | [paper-gigaworld-1-policy-evaluation](../entities/paper-gigaworld-1-policy-evaluation.md) | 系统比较世界模型与动作表示；强调长时动作忠实性。 |

## 关联页面

- [六路线技术地图](./embodied-wm-six-routes-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
