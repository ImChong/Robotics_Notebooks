---
type: entity
tags: [paper, loco-manip-contact-survey, humanoid, loco-manipulation, vla, optimal-control, locomotion, dfki]
status: complete
updated: 2026-07-03
arxiv: "2606.25591"
venue: "2026 · arXiv"
summary: "WOLF-VLA 用全身最优控制合成动态一致 loco 轨迹数据集，训练语言条件 VLA 生成人形运动策略，强调最优性、安全与接触一致性。"
related:
  - ../overview/loco-manip-contact-technology-map.md
  - ../overview/loco-manip-contact-category-05-vla-world-models.md
  - ../methods/vla.md
  - ../methods/model-predictive-control.md
  - ./paper-loco-manip-161-154-openhlm.md
sources:
  - ../../sources/papers/wolf_vla_arxiv_2606_25591.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# WOLF-VLA

**WOLF-VLA**（arXiv:2606.25591，DFKI 等）收录于 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) **05 VLA/WM 调用** 组：用 **全身最优控制数据工厂** 支撑 **语言条件 VLA** 人形运动。

## 一句话定义

**WOLF**（Whole-Body Humanoid Optimal Locomotion Framework）仅通过 **OCP 求解** 生成六类 locomotion 相关、动态可行且接触一致的全身轨迹；配合 ego 视觉与自然语言指令训练 **VLA**，从指令直接输出人形运动策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WOLF-VLA | Whole-Body Humanoid Optimal Locomotion Framework for VLA | 本文统一框架 |
| VLA | Vision-Language-Action | 多模态策略输出全身运动 |
| OC | Optimal Control | 轨迹合成保证最优性与约束 |
| OCP | Optimal Control Problem | 数据集生成核心 |
| MPC | Model Predictive Control | 与经典人形控制栈相关对照 |

## 为什么重要

- **动态一致数据：** 回应 VLA 在人形 loco 上缺乏 **物理可行示范** 的瓶颈。
- **最优性与安全：** 相对遥操作数据，OCP 轨迹编码能量/约束，利于可迁移训练。
- **策展对照：** 与 MotionWAM/HAIC 等同组，强调上层模型须落到 **带接触与动力学结构** 的全身接口。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 分组 | Loco-Manip 接触专题 · 05 VLA/WM 调用 |
| 原文题目 | WOLF-VLA: Whole-Body Humanoid Optimal Locomotion Framework for Vision-Language-Action Learning |
| 机构 | DFKI；University of Oldenburg 等 |
| 论文/项目 | <https://arxiv.org/abs/2606.25591> |

## 评测速览（索引级）

- **数据合成：** OCP 求解生成 **六类** locomotion 相关、动态可行且接触一致的全身轨迹。
- **策略验证：** 语言条件 VLA 从 ego 视觉 + 指令直接输出人形运动策略，强调最优性、安全与接触一致性。
- 定量指标以 arXiv:2606.25591 为准，索引级页暂不展开。

## 与其他页面的关系

- 分类 hub：[loco-manip-contact-category-05-vla-world-models.md](../overview/loco-manip-contact-category-05-vla-world-models.md)
- 姊妹：[OpenHLM](paper-loco-manip-161-154-openhlm.md)、[MotionWAM](paper-motionwam-humanoid-loco-manipulation-wam.md)

## 参考来源

- [wolf_vla_arxiv_2606_25591.md](../../sources/papers/wolf_vla_arxiv_2606_25591.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [VLA](../methods/vla.md)
- [模型预测控制](../methods/model-predictive-control.md)
