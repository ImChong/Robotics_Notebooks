---
type: entity
tags: [paper, world-models, survey-curated, embodied-wm-six-routes]
status: complete
updated: 2026-08-29
venue: curated
related:
  - ./paper-riemann-1.md
  - ../overview/embodied-wm-six-routes-technology-map.md
  - ../overview/embodied-wm-route-action.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
  - ../../sources/papers/riemann_1_0.md
summary: "Riemann-1.0 六路线策展入口（行动主导型）：先动作、后视觉后果。方法/数字/开源核查见 canonical 实体页 paper-riemann-1。"
---

# Riemann-1.0（六路线策展入口）

> **Canonical 实体页：** [Riemann-1.0（全因果自回归 World Action Model）](./paper-riemann-1.md)
>
> 本页只保留 [具身世界模型六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 的 **行动主导型** 导航坐标；方法、评测与开源状态以项目页 ingest 为准。

## 一句话定义

**六路线里的行动主导型样本：历史观测先生成动作，再条件预测视觉后果；真机可直接执行。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| WM | World Model | 根据状态与动作预测未来观测 |
| VLA | Vision-Language-Action | 反应式语义策略对照 |
| PSR | Progress Success Rate | 真机过程成功率（详见 canonical 页） |

## 为什么重要

- 综述把它放在 **行动主导型**：部署时未来建模与动作生成显式耦合，而不是只当画质生成器。
- 2026-08-29 已用官方项目页与技术报告 PDF 升格 [完整实体页](./paper-riemann-1.md)；本页不再重复数字。

## 核心信息

| 项 | 内容 |
|----|------|
| **路线** | 行动主导型 |
| **因果顺序** | 先 \(a_t\)，再 \(z_t\) |
| **详情** | [paper-riemann-1](./paper-riemann-1.md) |
| **开源** | **确认未开源**（见 canonical 页步骤 2.5） |

## 实验与评测

量化以 [Riemann-1.0 实体页](./paper-riemann-1.md) 的仿真 / 真机表为准（RoboCasa365 62.6%、真机均 85.0% SR）。本页不转抄，避免与项目页 ingest 双源漂移。

## 与其他工作对比

同路线邻接：[UWM](./paper-shenlan-wm-08-uwm.md)、[Cosmos Policy](./paper-shenlan-wm-11-cosmos-policy.md)、[World Tokens](./paper-world-tokens-inference-trimmed-wam.md)。对照时先问「生成分支在部署时是否还跑」，再问架构。完整对照表见 [canonical 页](./paper-riemann-1.md)。

## 结论

**读六路线时把它当作「动作优先因果」的导航钉；要数字、课程和开源边界，跳到 [paper-riemann-1](./paper-riemann-1.md)。**

1. 归入行动主导型看的是闭环职责，不是机构名。
2. 与 World Tokens「推理裁生成分支」对照时，Riemann 部署仍可滚动作条件视频。
3. 不要在本页堆 LIBERO / RoboTwin 饱和榜。

## 关联页面

- [Riemann-1.0 完整实体](./paper-riemann-1.md)
- [具身世界模型六路线技术地图](../overview/embodied-wm-six-routes-technology-map.md)
- [行动主导型 分类 hub](../overview/embodied-wm-route-action.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [六路线综述摘录](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)
- [Riemann-1.0 论文摘录](../../sources/papers/riemann_1_0.md)

## 推荐继续阅读

- [Riemann-1.0 项目页](https://riemann-dynamics.github.io/Riemann-1.0-Website)
- [六路线原文](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ)
