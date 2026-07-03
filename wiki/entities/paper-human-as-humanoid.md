---
type: entity
tags: [paper, loco-manip-contact-survey, humanoid, loco-manipulation, ego-exo, vla, data-pipeline]
status: complete
updated: 2026-07-03
venue: curated
summary: "Human-as-Humanoid 以 PrimeU 人形对齐本体将 Ego-Exo 人类视频转为 60-DoF 控制器对齐动作标签，PhysDex FK 感知训练；示范吞吐较遥操作高 4.8–7.2×，可无目标任务机器人示范零样本部署。"
related:
  - ../overview/loco-manip-contact-technology-map.md
  - ../overview/loco-manip-contact-category-01-contact-data.md
  - ../tasks/loco-manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/human_as_humanoid_zgc_2026.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Human-as-Humanoid

**Human-as-Humanoid** 收录于 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) **01 接触数据** 组：将 **Ego-Exo** 人类视频转为可训练的人形动作标签，降低 loco-manip 数据采集对机器人遥操作的依赖。

## 一句话定义

以 **PrimeU**（60-DoF 上身人形对齐本体）为底座，同步 ego-exo 视频经运动恢复与分阶段 IK 生成控制器对齐动作块，**PhysDex** 用 FK 感知监督保持腕/指尖任务空间几何；相对遥操作 **4.8–7.2×** 吞吐，支持无目标任务机器人示范的零样本部署。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | PhysDex 为 flow-matching DiT VLA |
| Ego-Exo | Egocentric + Exocentric | 第一视角输入 + 外视角运动恢复 |
| FK | Forward Kinematics | DS-HKC 可微 FK 约束腕/指尖几何 |
| IK | Inverse Kinematics | 分阶段 IK 生成关节动作块 |
| Loco-Manip | Loco-Manipulation | 本文聚焦高 DoF 上身操作数据，非纯行走 |

## 为什么重要

- **数据入口创新：** 把人类视频变成 **机器人可执行标签**，直接回应接触专题「带本体可执行性的交互数据从哪来」。
- ** embodiment 先行：** PrimeU 按成人男性操作比例设计，缩小重定向前的人-机形态 gap。
- **吞吐对比：** 采集链路约 **20 FPS** 近实时转换，显著快于全身遥操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 分组 | Loco-Manip 接触专题 · 01 接触数据 |
| 论文/项目 | <https://zgc-embodyai.github.io/Human-as-Humanoid/> |
| 机构 | ZGC EmbodyAI 等 |

## 与其他页面的关系

- 分类 hub：[loco-manip-contact-category-01-contact-data.md](../overview/loco-manip-contact-category-01-contact-data.md)
- 姊妹数据路线：[HumanoidUMI](paper-humanoidumi.md)、[VLK](paper-vlk-synthetic-loco-manipulation.md)

## 参考来源

- [human_as_humanoid_zgc_2026.md](../../sources/papers/human_as_humanoid_zgc_2026.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [接触五段链路技术地图](../overview/loco-manip-contact-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
