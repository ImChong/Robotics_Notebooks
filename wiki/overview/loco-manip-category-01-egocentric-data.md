---
type: overview
tags: [loco-manipulation, egocentric, category-hub, survey, vla]
status: complete
updated: 2026-06-14
summary: "Loco-Manip 8 篇周报 · 01 第一视角数据（2 篇）— 人类 ego 视频如何先补任务语义（Ego-Pi），再补全身动作先验（EgoPriMo）？"
related:
  - ./loco-manip-8-papers-technology-map.md
  - ./ego-9-papers-technology-map.md
  - ./loco-manip-category-02-synthetic-data.md
  - ../entities/paper-loco-manip-01-ego-pi.md
  - ../entities/paper-loco-manip-02-egoprimo.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md
  - ../../sources/papers/loco_manip_8_papers_catalog.md
---

# Loco-Manip 分类 01：第一视角数据

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 8 篇周报](https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ) 的 **01 第一视角数据** 分组；总地图见 [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与操作记录 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| SMPL | Skinned Multi-Person Linear Model | 参数化人体网格与姿态表示 |

## 核心问题

**人类第一视角如何进入人形 loco-manip 训练链？** 须区分 **任务语义**（顺序、物体、阶段）与 **身体动作**（走近、转身、全身协调）；避免把人类关节轨迹硬拷贝给机器人。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | Ego-Pi | [paper-loco-manip-01-ego-pi.md](../entities/paper-loco-manip-01-ego-pi.md) | [source](../../sources/papers/loco_manip_survey_01_ego_pi.md) |
| 02 | EgoPriMo | [paper-loco-manip-02-egoprimo.md](../entities/paper-loco-manip-02-egoprimo.md) | [source](../../sources/papers/loco_manip_survey_02_egoprimo.md) |

## 与 Ego 9 篇地图的分工

| 维度 | [Ego 9 篇](./ego-9-papers-technology-map.md) | 本组（Loco-Manip 周报） |
|------|---------------------------------------------|-------------------------|
| 范围 | 采集 → 人→机 → WM → Ego+Exo 全链路 | 聚焦 **loco-manip 训练入口** |
| 代表 | AoE、EgoMimic、WEM 等 | Ego-Pi（VLA 共微调）、EgoPriMo（全身 SMPL 生成） |

## 关联页面

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
- [生成与仿真数据](./loco-manip-category-02-synthetic-data.md)
- [VLA](../methods/vla.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_8_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)
- [loco_manip_8_papers_catalog.md](../../sources/papers/loco_manip_8_papers_catalog.md)

## 推荐继续阅读

- [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)
