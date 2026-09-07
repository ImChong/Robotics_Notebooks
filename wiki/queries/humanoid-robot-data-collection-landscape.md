---
type: query
tags: [data-collection, humanoid, teleoperation, investment, physical-ai]
status: complete
updated: 2026-09-07
summary: "Query：综合 LeoInAI Substack（2026-09-06）与 Scanford 论文，梳理人形/机器人训练数据采集六条范式及独立实体索引。"
related:
  - ../entities/paper-scanford-robot-powered-data-flywheel.md
  - ../entities/scanford.md
  - ../concepts/data-flywheel.md
  - ../tasks/teleoperation.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/leoinai_humanoid_robot_datacollection_2026-09-06.md
  - ../../sources/papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md
---

> **Query 产物**：本页由以下问题触发：「人形机器人训练数据怎么采？Substack 里提到的项目/论文各自是什么？」
> 综合来源：[LeoInAI Substack](../../sources/blogs/leoinai_humanoid_robot_datacollection_2026-09-06.md)、[Scanford 论文摘录](../../sources/papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md)

# Query：人形机器人数据采集产业地图

机器人训练数据 **不能** 像 LLM 那样纯爬文本：需要 **速度、姿态、力触、egocentric 视觉、接触滑移** 等物理量。LeoInAI Substack（2026-09-06）把当前赛道拆成多条并行范式；本页为 **独立实体索引**（每个项目/论文一条详情页，避免在本页重复技术细节）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 示范驱动的主流训练范式 |
| POV | Point of View | 第一人称 egocentric 视觉 |
| VLA | Vision-Language-Action | 语言条件策略；数据需对齐动作接口 |
| FM | Foundation Model | 飞轮中可被持续微调的基座模型 |
| QA | Quality Assurance | 采集商品化后的差异化：人类质检 |

## 六条采集范式（与实体页）

| 范式 | 代表节点 | 一句话 |
|------|----------|--------|
| **1. 野外机器人飞轮** | [RPDF / Scanford](../entities/paper-scanford-robot-powered-data-flywheel.md) · [Scanford 系统](../entities/scanford.md) | 机器人边干活边用任务结构自动标注，闭环微调 VLM |
| **2. VR/外骨骼遥操作** | [Boston Dynamics](../entities/boston-dynamics.md) · [Tesla Optimus](../entities/tesla-optimus.md) · [1X](../entities/1x-technologies.md) | 人戴 VR/追踪器，机器人镜像记录 onboard |
| **3. 可穿戴无机器人** | [Mimic U1](../entities/mimic-wearable-u1.md) · [Paxini](../entities/paxini.md) · [Rokoko](../entities/rokoko.md) | 手套/外骨骼/动捕，人在真场景操作 |
| **4. 服务换数据** | [Shift](../entities/shift-app-nyc.md) · [Figure Index](../entities/figure-ai.md) | 免费保洁或用户众包录像 |
| **5. 消费品侧传感** | [Dyson CameraJet](../entities/dyson-camerajet.md) | 高频日用品内置相机（机器人解读为潜在臂部统计源） |
| **6. 标注/地产层** | [Innodata](../entities/innodata.md) · [Appen](../entities/appen.md) · [Micro1](../entities/micro1.md) · [Brookfield](../entities/brookfield-physical-ai-data.md) | 动捕实验室、人类质检、物业场景房东 |
| **7. 世界模型平台（假想整合）** | [Cosmos 3](../entities/cosmos-3.md) | 端到端「采数–训模」垂直整合讨论参照 |

## 结论（可操作）

1. **没有单一「纯数据公司」标的** — 公开市场多是 [Innodata](../entities/innodata.md)/[Appen](../entities/appen.md) 等 **数据服务层**；机器人厂自己采数（[Figure](../entities/figure-ai.md)、[Tesla](../entities/tesla-optimus.md)）。
2. **采集商品化、质检稀缺** — Appen/Micro1 均强调 **人类判断哪些片段可学**；投资/选型应看 QA 能力而不只看摄像头数量。
3. **地产成为隐形基建** — [Paxini](../entities/paxini.md) 自建工厂 vs [Brookfield](../entities/brookfield-physical-ai-data.md) 型现有物业，是两种 **场景所有权** 模型。
4. **「回购机器人」可能是飞轮** — Substack 用 [RPDF](../entities/paper-scanford-robot-powered-data-flywheel.md) 解释：买回机器人为 **采真实环境数据**，不等同于无需求。
5. **学术可引用锚点** — 飞轮机制以 [arXiv:2511.19647](https://arxiv.org/abs/2511.19647) 为准；产业案例以各实体页与原始链接为准。

## 参考来源

- [LeoInAI Substack 归档](../../sources/blogs/leoinai_humanoid_robot_datacollection_2026-09-06.md)
- [Scanford 论文摘录](../../sources/papers/scanford_robot_powered_data_flywheel_arxiv_2511_19647.md)

## 关联页面

- [Data Flywheel](../concepts/data-flywheel.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Manipulation](../tasks/manipulation.md)

## 推荐继续阅读

- <https://leoinai.substack.com/p/invest-humanoid-robot-datacollection>
- <https://arxiv.org/abs/2511.19647>
