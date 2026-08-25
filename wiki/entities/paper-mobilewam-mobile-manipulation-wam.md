---
type: entity
tags: [paper, world-models, survey-curated, embodied-wm-six-routes]
status: complete
updated: 2026-08-25
venue: curated

related:
  - ../overview/embodied-wm-six-routes-technology-map.md
  - ../overview/embodied-wm-route-action.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md
summary: "MobileWAM（具身世界模型六路线专题）：从机械臂扩展到移动操作的 WAM。"
---

# MobileWAM

**MobileWAM** 收录于 [具身智能研究室 · 具身世界模型六路线综述](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) **行动主导型** 段。本页为知识库 **策展编译** 详情节点；量化指标以原文 PDF / 项目页为准。

## 一句话定义

从机械臂扩展到移动操作的 WAM。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 根据状态与动作预测未来观测/状态 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| MPC | Model Predictive Control | 滚动时域优化选动作 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 属于六路线 taxonomy 的 **行动主导型**：预测结果被用于 **部署时动作生成与未来建模显式耦合。**
- 与 [六路线技术地图](../overview/embodied-wm-six-routes-technology-map.md) 中同段工作对照阅读，避免与 WAM/评估/记忆类节点混淆。

## 核心信息

| 项 | 内容 |
|----|------|
| **路线** | 行动主导型 |
| **出处** | （策展文未给 arXiv；以原文为准） |
| **文内角色** | 从机械臂扩展到移动操作的 WAM。 |

## 结论

**MobileWAM 在六路线框架下的读法：先看预测输出被谁消费，再看是否改善真实行动——而非只看网络新旧或画面观感。**

- 归入 **行动主导型** 的判断标准是 **闭环职责**，不是模型架构标签。
- 与同路线邻接工作对照时，优先比较 **动作条件性、物理一致性与部署接口**。
- 细节数字与开源状态以原文为准；本页服务图谱导航与交叉引用。

## 实验与评测

- **本页无量化数字**：六路线综述只给出该工作在 taxonomy 中的定位，未转述实验表格；成功率、消融与实机协议以 [综述原文](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ) 指向的论文 / 项目页为准。
- **该路线该看的指标**：真机执行成功率、动作条件性（预测是否随动作改变）与推理时延。
- **综述的评价取向**：按文内判断「评价从画质转向行动效用」，读实验时先问预测是否改善了真实执行，再看画面观感（见 [六路线技术地图](../overview/embodied-wm-six-routes-technology-map.md)）。

## 与其他工作对比

- **同路线邻接工作**（综述 **行动主导型** 段）：[MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md)、[Riemann-1.0](./paper-riemann-1-causal-action-video-wam.md)。
- **对照要点**：MobileWAM 把 WAM 从 **固定臂** 扩到 **移动底盘 + 臂**，视点随本体移动；MotionWAM 面向 **人形全身 loco-manipulation**——两者的难点都在自由度与视点漂移，可直接对读。
- **跨路线区分**：行动主导型的闭环职责是「执行时未来建模与动作生成显式耦合」；与其他路线的分界是 **预测结果被谁消费**，不是模型架构或参数量。定量对照回到各自原文，本页不做跨论文数字拼接。

## 关联页面

- [具身世界模型六路线技术地图](../overview/embodied-wm-six-routes-technology-map.md)
- [行动主导型 分类 hub](../overview/embodied-wm-route-action.md)
- [Generative World Models](../methods/generative-world-models.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md](../../sources/blogs/wechat_embodied_ai_lab_wm_six_routes_survey_2026-08-25.md)

## 推荐继续阅读

- [具身世界模型六路线原文](https://mp.weixin.qq.com/s/mmIJRp9g6NqblMCjd9D5GQ)
