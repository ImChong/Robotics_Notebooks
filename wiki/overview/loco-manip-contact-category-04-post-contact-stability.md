---
type: overview
tags: [loco-manipulation, contact-rich, category-hub, survey, force-control, impedance-control]
status: complete
updated: 2026-07-03
summary: "Loco-Manip 接触专题 · 04 接触后稳定（7 篇）— 力自适应、阻抗、柔顺、负载摆动与强接触下身体如何持续？"
related:
  - ./loco-manip-contact-technology-map.md
  - ./loco-manip-contact-category-02-contact-representation.md
  - ./loco-manip-contact-category-05-vla-world-models.md
  - ../concepts/impedance-control.md
  - ../concepts/hybrid-force-position-control.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Loco-Manip 接触分类 04：接触后如何稳住

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) 的 **04 接触后稳定** 段；总地图见 [接触五段链路技术地图](./loco-manip-contact-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 全身协调；负载摆动须全身间接控制 |
| RL | Reinforcement Learning | WoCoCo 顺序接触分阶段学习 |
| MPC | Model Predictive Control | 部分力控/异构元控制路线 |
| HOI | Human-Object Interaction | 推门、擦拭、协作搬运等持续接触任务 |

## 核心问题

真机失败常发生在 **接触已建立之后**：手上力变、脚下支撑变、物体位置变，身体仍跟踪旧轨迹。**位置跟踪只是开始，持续接触才是真机难点。**

## 本组工作（7 篇）

| 工作 | Wiki 实体（复用） | 文内角色 |
|------|-------------------|----------|
| FALCON | [paper-loco-manip-161-109-falcon](../entities/paper-loco-manip-161-109-falcon.md) | 下肢稳定 + 上肢任务分工；3D 力课程 |
| HMC | [paper-loco-manip-161-039-hmc](../entities/paper-loco-manip-161-039-hmc.md) | 位置/阻抗/力位混合的异构元控制 |
| WoCoCo | [paper-loco-manip-161-116-wococo](../entities/paper-loco-manip-161-116-wococo.md) | 顺序接触分阶段全身控制 |
| GentleHumanoid | [paper-hrl-stack-37-gentlehumanoid](../entities/paper-hrl-stack-37-gentlehumanoid.md) | 上肢柔顺接入全身跟踪 |
| CHIP | [paper-hrl-stack-36-chip](../entities/paper-hrl-stack-36-chip.md) | 事后扰动学习自适应柔顺 |
| HOIST | [paper-motion-cerebellum-hoist](../entities/paper-motion-cerebellum-hoist.md) | 悬挂负载摆动；欠驱动间接全身控制 |
| Thor | [paper-hrl-stack-42-thor](../entities/paper-hrl-stack-42-thor.md) | 强接触环境受力后全身反应 |

## 策展判断

- **FALCON / HMC：** 力自适应与 **控制模式切换** 偏工程、贴近真机接触。
- **WoCoCo：** 与 OmniContact 接触流共享 **接触有阶段、有切换** 的问题意识。
- **GentleHumanoid / CHIP：** 过硬位置跟踪易把接触变成 **冲击**。
- **HOIST / Thor：** 补 **接触建立后继续与世界交换力** 的能力。

## 关联页面

- [阻抗控制](../concepts/impedance-control.md)
- [力位混合控制](../concepts/hybrid-force-position-control.md)
- [运动小脑 · I 柔顺接触](./motion-cerebellum-category-09-compliance-contact.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [触觉阻抗控制](../methods/tactile-impedance-control.md)
- [接触力控制带宽](../concepts/contact-force-loop-bandwidth.md)
