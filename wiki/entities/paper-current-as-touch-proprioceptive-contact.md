---
type: entity
tags:
  - paper
  - tactile
  - proprioception
  - compliance
  - dexterous-manipulation
  - contact-rich
status: complete
updated: 2026-07-11
arxiv: "2607.03529"
related:
  - ../overview/wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-vt-wam-visuotactile-contact-rich.md
  - ../entities/paper-deform360-deformable-visuotactile-dataset.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "Current as Touch（arXiv:2607.03529）：从电机电流与关节状态学习柔顺参考位置（CRP），标准 PD 位置控制产生合适抓取力；无需外置触觉/力传感器，覆盖纸杯、擦拭、动态负载等任务。"
---

# Current as Touch（Proprioceptive Contact Feedback）

**Current as Touch**（arXiv:2607.03529，[项目页](https://cat.chenyangma.com/)）——见策展导读与一手论文。

## 一句话定义

**用电机电流作本体触觉：预测柔顺参考关节位置，位置 PD 间接实现力适应**——低成本的接触反馈源。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CRP | Compliance Reference Position | PD 控制器的柔顺目标关节位置 |
| PD | Proportional-Derivative | 主流位置控制接口 |
| IoT | — | （无）强调无需额外传感硬件 |

## 为什么重要

- 拓展策展文 **四类接触信号** 中的 **电机电流** 路线。
- 与硬件触觉（VT-WAM）互补：**执行器自带信号** 也可建模接触。
- 位置式接口兼容主流遥操作与策略学习管线。

## 核心结构

| 设计 | 作用 |
|------|------|
| 电机电流 + 关节状态 | 接触力、阻力、稳定性代理信号 |
| CRP 预测 | 柔顺目标位置，非直接力矩命令 |
| 标准 PD | 位置误差产生合适交互力 |

## 实验要点（策展口径）

多灵巧手、纸杯/薄片/擦拭/动态负载等接触丰富任务（见项目页）。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-02-contact-modeling](../overview/wm-action-consequence-category-02-contact-modeling.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.03529](https://arxiv.org/abs/2607.03529) — 一手论文
- [https://cat.chenyangma.com/](https://cat.chenyangma.com/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.03529](https://arxiv.org/abs/2607.03529)
