---
type: overview
tags: [loco-manipulation, contact-rich, category-hub, survey, generative-models, sim2real]
status: complete
updated: 2026-07-22
summary: "Loco-Manip 接触专题 · 03 生成式补数（7 篇）— 生成视频、3D 资产、高斯场景与仿真能否提供可训练接触轨迹？"
related:
  - ./loco-manip-contact-technology-map.md
  - ./loco-manip-contact-category-01-contact-data.md
  - ./loco-manip-8-papers-technology-map.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Loco-Manip 接触分类 03：生成式路线补数据

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) 的 **03 生成式补数** 段；总地图见 [接触五段链路技术地图](./loco-manip-contact-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Human-Object Interaction | 人-物交互；GenHOI 从生成视频恢复 |
| VLA | Vision-Language-Action | LEGS 等用高斯场景生成 VLA 训练数据 |
| Sim2Real | Simulation to Reality | OASIS 仿真遥操作 → 域随机化 → 真机 |
| RL | Reinforcement Learning | Humanoid-DART 扩散轨迹 + RL 跟踪 |

## 核心问题

只靠真实示范与遥操作，**接触数据很难扩到足够大**。生成路线补的是 **长尾接触场景**；价值不在画面像不像，而在能否提供 **可恢复、可跟踪、可训练** 的接触轨迹，并做 **物理一致性验证**。

## 本组工作（7 篇）

| 工作 | Wiki 实体（复用） | 文内角色 |
|------|-------------------|----------|
| GenHOI | [paper-loco-manip-03-genhoi](../entities/paper-loco-manip-03-genhoi.md) | 生成视频 → 交互轨迹 → 零样本执行 |
| Imagine2Real | [paper-imagine2real-zero-shot-hoi](../entities/paper-imagine2real-zero-shot-hoi.md) | 稀疏关键点 + 零样本 HOI，弱化精细重定向 |
| GRAIL | [paper-grail](../entities/paper-grail.md) | 3D 资产 + 视频先验生成 loco-manip 数据 |
| Humanoid-DART | [paper-humanoid-dart](../entities/paper-humanoid-dart.md) | 扩散生成轨迹 + RL 跟踪，扩大目标空间 |
| LEGS | [paper-legs-embodied-gaussian-splatting-vla](../entities/paper-legs-embodied-gaussian-splatting-vla.md) | 高斯场景重建 → 无遥操作 VLA 数据 |
| OASIS | [paper-loco-manip-04-oasis](../entities/paper-loco-manip-04-oasis.md) | 仿真资产/遥操作/域随机化 → 真机部署 |
| SIMPLE | [paper-loco-manip-161-075-simple](../entities/paper-loco-manip-161-075-simple.md) | 接触动力学 + 视觉渲染 + 任务资产一体化 |

## 策展判断

- **GenHOI / Imagine2Real：** 视频生成先验驱动 HOI；须验证脚底打滑、物体支撑、重心边界。
- **GRAIL / Humanoid-DART：** 数字资产或扩散 **扩大可训练目标空间**。
- **LEGS / OASIS / SIMPLE：** **基础设施侧**——场景外观、仿真闭环与接触动力学采集。

## 关联页面

- [Loco-Manip 8 篇 · 生成与仿真](./loco-manip-category-02-synthetic-data.md)
- [GRAIL 数据集](../entities/grail-locomanipulation-dataset.md)
- [LEGS VLA 实体](../entities/paper-legs-embodied-gaussian-splatting-vla.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [机器人世界模型训练闭环](./robot-world-models-training-loop-taxonomy.md)
- [GenHOI vs SimGenHOI](../entities/paper-loco-manip-03-genhoi.md) — 勿与 SimGenHOI 合并节点
