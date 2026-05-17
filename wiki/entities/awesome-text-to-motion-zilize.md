---
type: entity
tags: [repo, curated-list, text-to-motion, human-motion, smpl, dataset, survey]
status: complete
updated: 2026-05-17
related:
  - ../methods/diffusion-motion-generation.md
  - ../methods/hy-motion-1.md
  - ../methods/genmo.md
  - ../methods/motion-retargeting-gmr.md
  - ../methods/skeleton-action-recognition.md
sources:
  - ../../sources/repos/zilize-awesome-text-to-motion.md
summary: "Zilize 维护的 awesome-text-to-motion：文本驱动单人人体运动生成的综述、数据集与模型精选清单，配套 GitHub Pages 交互可视化；范围刻意排除人–物/场景交互，便于与机器人重定向与控制文献对照阅读。"
---

# Awesome Text-to-Motion（Zilize 精选集）

**Awesome Text-to-Motion**（GitHub 仓名 `awesome-text-to-motion`）是一份 **文本驱动人体运动生成** 的 curated 列表：按 **Surveys / Datasets / Models** 组织条目，元数据维护在 `data/arxiv.csv` 与 `data/without-arxiv.json`，并通过 [项目主页](https://zilize.github.io/awesome-text-to-motion/) 提供 **Plotly 交互图与统计视图**。

## 一句话定义

面向 **单人、无 HOI** 设定的人体 **文本→3D 运动** 文献与资源的 **可筛选索引入口**（与 broader 交互式人体运动生成综述互补）。

## 为什么重要？

- **降低检索成本**：HumanML3D、Motion-X、KIT-ML 等基准与 GENMO、MotionGPT 系列等模型在同一套标签体系下并列，适合快速建立子领域地图。
- **边界清晰**：README 明确聚焦 **single-person** 且 **不含 human-object / scene interaction**，与机器人侧常关心的 **接触、搬运、场景几何** 任务形成自然对照，避免把两类问题混读。
- **与 Sim2Real 接口相邻**：人体 SMPL 系轨迹常作为 [Motion Retargeting](../methods/motion-retargeting-gmr.md) 的上游；本列表有助于定位「高质量文本条件人体先验」来自哪条研究线。

## 核心结构（怎么读）

| 区块 | 内容侧重 |
|------|-----------|
| Surveys | 生成范式总览、多模态 LLM 路线、文本驱动综述与 TPAMI 长综述 |
| Datasets | 从经典 HumanML3D / KIT-ML 到百万级、细粒度标注、长文本扩展等 |
| Models | 扩散、自回归、检索增强、偏好对齐、层级行为控制等实现条目 |

## 局限与使用注意

- **非机器人控制论文库**：条目主体在 **计算机视觉 / 图形学** 语境下的 3D 人体运动，落地到足式/人形控制需额外考虑接触、平衡与执行器约束。
- **清单滞后**：任何 awesome 列表都依赖 PR 维护；关键论文应以 arXiv / 官方仓为准，本页仅作拓扑索引。

## 关联页面

- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md) — 机器人域扩散轨迹生成；可与人体 T2M 扩散方法对照
- [HY-Motion 1.0](../methods/hy-motion-1.md) — 腾讯混元开源的十亿级 DiT+流匹配文本→SMPL-H 运动系列（arXiv:2512.23464）
- [GENMO（统一人体运动估计与生成）](../methods/genmo.md) — 本列表收录的代表性人体扩散「通才」模型（GEM 发布名）
- [General Motion Retargeting（GMR）](../methods/motion-retargeting-gmr.md) — 人体运动→机器人骨架的常见工程落点
- [Skeleton-based Action Recognition](../methods/skeleton-action-recognition.md) — HumanML3D 等在识别/异构基准中的使用语境

## 参考来源

- [sources/repos/zilize-awesome-text-to-motion.md](../../sources/repos/zilize-awesome-text-to-motion.md)

## 推荐继续阅读

- [Awesome Text-to Motion 项目主页（交互图表）](https://zilize.github.io/awesome-text-to-motion/)
- [GitHub 仓库 README（Surveys/Datasets/Models 全文）](https://github.com/Zilize/awesome-text-to-motion/blob/master/README.md)
- [Text-driven Motion Generation: Overview, Challenges and Directions](https://arxiv.org/abs/2505.09379)（列表内综述示例）
