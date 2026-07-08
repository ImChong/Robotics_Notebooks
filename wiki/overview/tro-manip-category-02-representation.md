---
type: overview
tags: [manipulation, category-hub, survey, representation, se3-equivariant, dexterous-manipulation]
status: complete
updated: 2026-07-08
summary: "T-RO 5 篇精选 · 02 三维与手物表征（2 篇）— SE(3) 等变规范化与 DexRep 手物几何如何提升操作 OOD 泛化？"
related:
  - ./tro-manip-5-papers-technology-map.md
  - ./tro-manip-category-01-data-scaling.md
  - ./tro-manip-category-03-video-pretraining.md
  - ../entities/paper-tro-manip-02-canonical-policy.md
  - ../entities/paper-tro-manip-03-dexrepnet-plus-plus.md
sources:
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# T-RO 分类 02：三维与手物表征

> **图谱分类节点**：对应 T-RO 2026 操作学习精选的 **02 三维与手物表征** 分组（2 篇）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SE(3) | Special Euclidean Group in 3D | 三维刚体旋转+平移变换群 |
| DRL | Deep Reinforcement Learning | 深度强化学习 |
| HOI | Hand-Object Interaction | 手-物交互与接触几何 |

## 核心问题

**如何用几何对称与手物空间关系，而非记忆特定姿态/视角？** 平行夹爪场景侧重 **SE(3) 等变点云策略**；灵巧手场景侧重 **指节-物体表面相对几何**。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 02 | Canonical Policy | [paper-tro-manip-02-canonical-policy.md](../entities/paper-tro-manip-02-canonical-policy.md) | [source](../../sources/papers/tro_manip_survey_02_canonical_policy.md) |
| 03 | DexRepNet++ | [paper-tro-manip-03-dexrepnet-plus-plus.md](../entities/paper-tro-manip-03-dexrepnet-plus-plus.md) | [source](../../sources/papers/tro_manip_survey_03_dexrepnet_plus_plus.md) |

## 关联页面

- [T-RO 5 篇技术地图](./tro-manip-5-papers-technology-map.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Diffusion Policy](../methods/diffusion-policy.md)

## 参考来源

- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- [tro_manip_5_papers_catalog.md](../../sources/papers/tro_manip_5_papers_catalog.md)

## 推荐继续阅读

- [灵巧操作数据管线与 RL 基建](../queries/dexterous-manipulation-data-pipeline.md)
