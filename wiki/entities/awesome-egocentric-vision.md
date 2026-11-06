---
type: entity
tags: [curated-list, egocentric, ego-vision, vla, world-models, hand-object-interaction, wearable]
status: complete
updated: 2026-11-04
related:
  - ../overview/sun-awesome-ego-technology-map.md
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-01-data-collection.md
  - ../overview/ego-category-02-human-to-robot.md
  - ../overview/ego-category-03-world-models.md
  - ../overview/ego-category-04-ego-exo-fusion.md
  - ./paper-ego4d.md
  - ../methods/egoscale.md
  - ./paper-egophi.md
  - ../methods/vla.md
  - ./awesome-world-models.md
  - ./awesome-touch.md
original:
  - ../../sources/repos/awesome_egocentric_vision_readme_reading_sun254667.md
sources:
  - ../../sources/repos/awesome-egocentric-vision.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
summary: "sun254667 维护的 Awesome Egocentric Vision：第一人称视觉与具身 AI 的论文/数据集/工具精选集；清单里的论文已拆成站内可点开的一页，并有按分组浏览的技术地图。"
---

# Awesome Egocentric Vision（sun254667 精选集）

**Awesome Egocentric Vision**（GitHub：[`sun254667/awesome-egocentric-vision`](https://github.com/sun254667/awesome-egocentric-vision)）是一份 **第一人称视觉与具身 AI** 的 curated 列表：从经典动作识别到 Ego-VLA / Ego World Models，并单独收录智能眼镜部署与隐私安全方向。

## 一句话定义

面向 **Egocentric（第一人称）感知 → 具身策略** 的学习友好型资源索引（论文、数据集、基准、仿真器与工具链）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物细粒度交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略（含 Ego-VLA） |
| VLM | Vision-Language Model | 多模态推理骨干 |
| Ego-Exo | Egocentric–Exocentric | 第一/第三人称协同学习 |

## 为什么重要

- **Ego 是具身数据规模化入口**：视线、手、遮挡与临场决策同时被记录；清单把「采什么、学什么、怎么进 WM」摊开。
- **与站内 Ego 地图互补**：[Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md) 给四类问题坐标；本清单给 **全谱近期文献与数据集表**。
- **交叉世界模型与操作**：§2.7 Ego-VLA、§2.8 Ego WM 可挂到 [VLA](../methods/vla.md) 与 [Awesome World Models](./awesome-world-models.md)。

## 站内怎么读

- **技术地图：** [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- **清单目录：** [sun_awesome_ego_catalog.md](../../sources/papers/sun_awesome_ego_catalog.md)
- 清单里的论文若站内已有深读页，会直接指向那一页。

## 核心结构（怎么读）

| 区块 | 内容侧重 |
|------|----------|
| Surveys | 任务四象限、程序化助手、Ego–Exo 协同、领域展望 |
| Core Topics | 动作识别/预测、HOI、Ego-VLM、长视频、注视、3D/4D、Ego-VLA、Ego WM、助手、隐私 |
| Datasets & Simulators | Ego4D / EPIC-KITCHENS 等锚点 + 细粒度/VLA/世界仿真器 |
| Smart Glasses | 可穿戴专用基准与部署技术 |

## 核心原理

清单的组织逻辑是 **「任务 → 数据 → 迁移」**（据 README 结构归纳）：2.1–2.6 为感知任务（动作、手物、VLM、长视频、注视、3D/4D），2.7–2.8 把感知接到策略与世界模型，2.9–2.10 覆盖助手与隐私。README 末尾「关键挑战」给出四个通用难点：视野窄/自遮挡/抖动、数据稀缺与隐私、跨人分布偏移、超长序列建模；Ego-VLA 一栏另列人机具身差距与动作空间错位。

## 工程实践

- **机器人方向优先读**：§2.7 Ego-VLA（如 EgoScale 的 2 万+ 小时带动作标注人类视频、HumanScale、EgoBridge）、§2.8 Ego WM，以及数据集表中的 EgoDex、EgoVerse、HOI4D。
- **选型锚点**：通用预训练用 Ego4D / EPIC-KITCHENS-100 / Ego-Exo4D；灵巧手与触觉看 EgoDex、EgoTouch；README 数据集表给出规模，使用前以官方页核验许可。
- **落地到站内**：先看[技术地图](../overview/sun-awesome-ego-technology-map.md)定位分组，再点进具体论文页。

## 局限与使用注意

- **非机器人控制论文库主体**：大量条目偏 CV 语境；落到真机策略需额外看重定向、接触与执行器约束。
- **与其他 Awesome-Egocentric 并存**：README 链到 EgoAlpha / Sid2697 等列表；选型时注意维护活跃度与收录边界。
- **清单滞后**：以 arXiv / 官方数据集页为准核验链接与许可。

## 关联页面

- [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) — 清单论文逐条可点开
- [Ego 技术地图：9 篇论文的四类问题视角](../overview/ego-9-papers-technology-map.md)
- [Ego 分类 01–04](../overview/ego-category-01-data-collection.md) — 采集 / 人→机 / WM / Ego+Exo
- [Ego4D](./paper-ego4d.md) — 大规模第一人称数据集锚点
- [EgoScale](../methods/egoscale.md) — 大规模 ego manipulation 标注与迁移
- [EgoPHI](./paper-egophi.md) — ego 视觉 mesh 级接触+力估计（ECCV 2026，arXiv:2608.13014）
- [VLA](../methods/vla.md)
- [Awesome World Models](./awesome-world-models.md) / [Awesome Touch](./awesome-touch.md)

## 参考来源

- [sources/repos/awesome-egocentric-vision.md](../../sources/repos/awesome-egocentric-vision.md)
- [sources/papers/sun_awesome_ego_catalog.md](../../sources/papers/sun_awesome_ego_catalog.md)

## 推荐继续阅读

- [GitHub 仓库 README](https://github.com/sun254667/awesome-egocentric-vision)
- [中文 README](https://github.com/sun254667/awesome-egocentric-vision/blob/main/README.zh-CN.md)
- [Ego4D 官网](https://ego4d-data.org/)
