---
type: overview
tags: [loco-manipulation, synthetic-data, simulation, category-hub, survey]
status: complete
updated: 2026-07-22
summary: "Loco-Manip 8 篇周报 · 02 生成与仿真数据（2 篇）— 生成视频与仿真 teleop 能否承担更重的可执行 loco-manip 数据生产（GenHOI、OASIS）？"
related:
  - ./loco-manip-8-papers-technology-map.md
  - ./loco-manip-category-01-egocentric-data.md
  - ./loco-manip-category-03-command-controller.md
  - ../entities/paper-loco-manip-03-genhoi.md
  - ../entities/paper-loco-manip-04-oasis.md
  - ../entities/paper-legs-embodied-gaussian-splatting-vla.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md
  - ../../sources/papers/loco_manip_8_papers_catalog.md
---

# Loco-Manip 分类 02：生成与仿真数据

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 8 篇周报](https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ) 的 **02 生成与仿真数据** 分组；总地图见 [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Human-Object Interaction | 人-物交互，含接触与物体运动 |
| DR | Domain Randomization | 域随机化，扩大仿真训练分布 |
| Sim2Real | Simulation to Reality | 仿真策略迁移到真实机器人 |

## 核心问题

**当真机 teleop 覆盖不足时，生成视频与仿真能否产出可执行 loco-manip 数据？** 生成侧重 **交互线索**（接触、物体运动）；仿真侧重 **批量轨迹 + 域随机化** 与零样本部署。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 03 | GenHOI | [paper-loco-manip-03-genhoi.md](../entities/paper-loco-manip-03-genhoi.md) | [source](../../sources/papers/loco_manip_survey_03_genhoi.md) |
| 04 | OASIS | [paper-loco-manip-04-oasis.md](../entities/paper-loco-manip-04-oasis.md) | [source](../../sources/papers/loco_manip_survey_04_oasis.md) |

## 常见误区

- **GenHOI ≠ SimGenHOI**：前者 arXiv:2606.12995（生成视频→接触线索）；后者为 Paper Notebooks 另一条 **SimGenHOI** 待深读条目，**勿合并节点**。
- **「仿真一定不如真机」**：OASIS 叙事强调 **覆盖度** 可补真实性不足（任务与资产质量依赖）。

## 关联页面

- [LEGS（3DGS VLA 数据工厂）](../entities/paper-legs-embodied-gaussian-splatting-vla.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [命令空间与控制器](./loco-manip-category-03-command-controller.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_8_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)
- [loco_manip_8_papers_catalog.md](../../sources/papers/loco_manip_8_papers_catalog.md)

## 推荐继续阅读

- [LEGS 微信策展](../../sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md)
