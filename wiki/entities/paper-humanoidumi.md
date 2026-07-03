---
type: entity
tags: [paper, loco-manip-contact-survey, humanoid, loco-manipulation, teleoperation, umi, data-pipeline, baai, unitree-g1]
status: complete
updated: 2026-07-03
arxiv: "2606.27239"
venue: "2026 · arXiv"
summary: "HumanoidUMI 用 PICO VR + UMI 式夹爪无机器人采集稀疏关键点与腕部视角，高层策略 + Spatial Keypoint Retargeting + 全身控制器在 G1 五类真机 loco-manip 任务验证。"
related:
  - ../overview/loco-manip-contact-technology-map.md
  - ../overview/loco-manip-contact-category-01-contact-data.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ./paper-loco-manip-07-wt-umi.md
sources:
  - ../../sources/papers/humanoidumi_arxiv_2606_27239.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# HumanoidUMI

**HumanoidUMI**（arXiv:2606.27239，BAAI）收录于 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) **01 接触数据** 组：将 **UMI 式无机器人示范** 扩展到 **人形全身** loco-manipulation。

## 一句话定义

轻量 **PICO VR + UMI 夹爪** 采集骨盆、双手 TCP、双脚等稀疏关键点与腕部 fisheye 图像；高层 visuomotor 策略预测关键点运动，经 **Spatial Keypoint Retargeting** 转为机器人全身参考，由学习的全身控制器在 **Unitree G1** 上执行。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UMI | Universal Manipulation Interface | 便携腕部-centric 示范范式 |
| WBC | Whole-Body Control | 低层执行 retarget 后全身参考 |
| Loco-Manip | Loco-Manipulation | 走、弯腰、投掷等与操作耦合的全身任务 |
| VR | Virtual Reality | PICO 4 追踪与关键点采集 |
| TCP | Tool Center Point | 夹爪工具中心点轨迹 |

## 为什么重要

- **降低采集门槛：** 无需机器人本体即可采全身意图，比全身遥操作更高效。
- **显式跨本体桥：** 稀疏关键点保留任务相关几何，为 human→humanoid 转移提供接口。
- **与 WT-UMI 对照：** 同属 UMI 谱系，HumanoidUMI 偏 **无机器人采集**，WT-UMI 偏 **触觉力全身遥操作**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 分组 | Loco-Manip 接触专题 · 01 接触数据 |
| 原文题目 | HumanoidUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation |
| 机构 | BAAI |
| 论文/项目 | <https://arxiv.org/abs/2606.27239> · <https://baai-aether.github.io/HumanoidUMI> |

## 与其他页面的关系

- 分类 hub：[loco-manip-contact-category-01-contact-data.md](../overview/loco-manip-contact-category-01-contact-data.md)
- 触觉对照：[WT-UMI](paper-loco-manip-07-wt-umi.md)

## 参考来源

- [humanoidumi_arxiv_2606_27239.md](../../sources/papers/humanoidumi_arxiv_2606_27239.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [接触五段链路技术地图](../overview/loco-manip-contact-technology-map.md)
- [Teleoperation 任务页](../tasks/teleoperation.md)
